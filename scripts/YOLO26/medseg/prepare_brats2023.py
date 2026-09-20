#!/usr/bin/env python3
"""Patient-disjoint BraTS2023 -> 2D YOLO segmentation, with original binary GT.

RGB channels are t1c, t2f, t2w. Foreground is the union of all nonzero labels.
All slices (including negatives) are retained unless an explicit smoke stride is used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

import cv2
import nibabel as nib
import numpy as np
import yaml

MODALITIES = ("t1c", "t2f", "t2w")
SPLITS = ("train", "val", "test")


def discover(src: Path) -> tuple[list[dict], dict]:
    """Use fixed MEN cases in place of originals, never as additional samples."""
    records = {}
    unlabeled = []
    replacements = []
    folders = sorted((p for p in src.iterdir() if p.is_dir()),
                     key=lambda p: ("FIX" in p.name.upper(), p.name))
    for folder in folders:
        for case in sorted(folder.glob("BraTS-*")):
            if not case.is_dir():
                continue
            cid = case.name
            parts = cid.split("-")
            if len(parts) != 4 or parts[1] not in ("GLI", "MEN", "PED"):
                raise ValueError(f"Unexpected case ID: {cid}")
            paths = {m: case / f"{cid}-{m}.nii.gz" for m in (*MODALITIES, "t1n")}
            missing = [str(p) for p in paths.values() if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete MRI case {cid}: {missing}")
            seg = case / f"{cid}-seg.nii.gz"
            rec = dict(case_id=cid, patient_id=cid.rsplit("-", 1)[0],
                       domain=parts[1], directory=str(case.resolve()))
            if not seg.is_file():
                unlabeled.append(rec)
                continue
            if cid in records:
                if "FIX" not in folder.name.upper():
                    raise ValueError(f"Unexpected duplicate labeled case: {cid}")
                replacements.append(cid)
            records[cid] = rec
    if not records:
        raise ValueError(f"No labeled BraTS cases under {src}")
    labeled_patients = {r["patient_id"] for r in records.values()}
    return sorted(records.values(), key=lambda r: r["case_id"]), {
        "labeled_cases": len(records),
        "labeled_patients": len(labeled_patients),
        "cases_by_domain": dict(Counter(r["domain"] for r in records.values())),
        "unlabeled_cases_by_domain": dict(Counter(r["domain"] for r in unlabeled)),
        "unlabeled_cases": unlabeled,
        "unlabeled_patients_also_labeled": sorted(labeled_patients & {r["patient_id"] for r in unlabeled}),
        "men_fixed_replacements": replacements,
    }


def split_patients(records: list[dict], seed: int, val_fraction: float,
                   test_fraction: float) -> list[dict]:
    if min(val_fraction, test_fraction) <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Need positive val/test fractions with sum < 1")
    assignments = {}
    for domain in sorted({r["domain"] for r in records}):
        patients = sorted({r["patient_id"] for r in records if r["domain"] == domain})
        random.Random(f"{seed}:{domain}").shuffle(patients)
        nval = max(1, round(len(patients) * val_fraction))
        ntest = max(1, round(len(patients) * test_fraction))
        if nval + ntest >= len(patients):
            raise ValueError(f"Too few patients in {domain} for three splits")
        for i, patient in enumerate(patients):
            assignments[patient] = "val" if i < nval else "test" if i < nval + ntest else "train"
    return [dict(r, split=assignments[r["patient_id"]]) for r in records]


def normalize(volume: np.ndarray) -> np.ndarray:
    if not np.isfinite(volume).all():
        raise ValueError("Nonfinite MRI intensities")
    nonzero = volume[volume != 0]
    if not nonzero.size:
        return np.zeros(volume.shape, dtype=np.uint8)
    lo, hi = np.percentile(nonzero, (0.5, 99.5))
    if hi <= lo:
        raise ValueError("Constant nonzero MRI intensities")
    normalized = np.clip((volume - lo) / (hi - lo), 0, 1)
    normalized[volume == 0] = 0
    return np.rint(normalized * 255).astype(np.uint8)


def polygons(mask: np.ndarray) -> tuple[list[str], int]:
    """External contours approximate YOLO targets; exact masks remain the evaluation GT."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows, omitted = [], 0
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) <= 0:
            omitted += 1
            continue
        xy = contour.reshape(-1, 2).astype(float) / np.array([w, h])
        rows.append("0 " + " ".join(f"{v:.7f}" for v in xy.ravel()))
    return rows, omitted


def export_case(record: dict, dst: Path, stride: int) -> dict:
    cid, split = record["case_id"], record["split"]
    case = Path(record["directory"])
    seg_img = nib.load(case / f"{cid}-seg.nii.gz")
    seg = np.asarray(seg_img.dataobj)
    if seg.ndim != 3 or not np.isfinite(seg).all() or (seg < 0).any() or not np.equal(seg, np.rint(seg)).all():
        raise ValueError(f"Invalid segmentation in {cid}")
    channels = []
    for modality in MODALITIES:
        img = nib.load(case / f"{cid}-{modality}.nii.gz")
        if img.shape != seg.shape or not np.allclose(img.affine, seg_img.affine, atol=1e-4):
            raise ValueError(f"MRI/mask geometry mismatch: {cid} {modality}")
        channels.append(normalize(img.get_fdata(dtype=np.float32)))
    # Transpose x,y into image row=y,column=x, identical for all channels and GT.
    zs = list(range(0, seg.shape[2], stride))
    positives = omitted = unrepresentable = 0
    for z in zs:
        stem = f"{cid}_z{z:03d}"
        rgb = np.stack([v[:, :, z].T for v in channels], axis=-1)
        mask = (seg[:, :, z].T > 0).astype(np.uint8)
        rows, small = polygons(mask)
        positives += bool(mask.any())
        omitted += small
        unrepresentable += bool(mask.any() and not rows)
        if not cv2.imwrite(str(dst / "images" / split / f"{stem}.png"), rgb[:, :, ::-1]):
            raise IOError(f"Failed to write image {stem}")
        if not cv2.imwrite(str(dst / "gt_masks" / split / f"{stem}.png"), mask * 255):
            raise IOError(f"Failed to write mask {stem}")
        (dst / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
    return dict(record, shape=list(seg.shape), spacing_mm=list(map(float, seg_img.header.get_zooms())),
                affine=seg_img.affine.tolist(), labels=list(map(int, np.unique(seg))),
                slices=zs, positive_slices=int(positives), negative_slices=len(zs)-int(positives),
                omitted_degenerate_contours=omitted, positive_slices_without_polygon=unrepresentable)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("dataset/BraTS2023"))
    ap.add_argument("--dst", type=Path, default=Path("dataset/BraTS2023-YOLO26"))
    ap.add_argument("--domains", nargs="+", choices=("GLI", "MEN", "PED"), default=["GLI", "MEN", "PED"])
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--max-cases-per-domain-split", type=int, default=0, help="Smoke subset, 0=all")
    ap.add_argument("--slice-stride", type=int, default=1, help="Smoke only; full evaluation requires 1")
    args = ap.parse_args()
    if args.slice_stride < 1 or args.max_cases_per_domain_split < 0:
        ap.error("stride must be >=1 and case limit >=0")
    dst = args.dst.resolve()
    if dst.exists() and any(dst.iterdir()):
        raise FileExistsError(f"Refusing to mix outputs in nonempty {dst}; choose a new --dst")
    records, audit = discover(args.src.resolve())
    records = [r for r in records if r["domain"] in args.domains]
    records = split_patients(records, args.seed, args.val_fraction, args.test_fraction)
    counts = {}
    for domain in args.domains:
        counts[domain] = {s: {"cases": sum(r["domain"] == domain and r["split"] == s for r in records),
                              "patients": len({r["patient_id"] for r in records if r["domain"] == domain and r["split"] == s})}
                          for s in SPLITS}
    audit.update(seed=args.seed, split_counts=counts)
    if args.max_cases_per_domain_split:
        selected = []
        for domain in args.domains:
            for split in SPLITS:
                subset = [r for r in records if r["domain"] == domain and r["split"] == split]
                random.Random(f"{args.seed}:{domain}:{split}:smoke").shuffle(subset)
                selected.extend(subset[:args.max_cases_per_domain_split])
        records = sorted(selected, key=lambda r: r["case_id"])
    manifest = dict(protocol="binary nonzero-label union; 2D axial; patient-disjoint by ID prefix",
                    rgb_modalities=MODALITIES, normalization="per-volume nonzero p0.5-p99.5; background=0",
                    seed=args.seed, val_fraction=args.val_fraction, test_fraction=args.test_fraction,
                    smoke_subset=bool(args.max_cases_per_domain_split or args.slice_stride != 1),
                    slice_stride=args.slice_stride, prepared=False, cases=records, audit=audit)
    digest_payload = [(r["case_id"], r["patient_id"], r["split"]) for r in records]
    manifest["split_sha256"] = hashlib.sha256(json.dumps(digest_payload).encode()).hexdigest()
    dst.mkdir(parents=True, exist_ok=True)
    manifest_path = dst / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "unlabeled_cases"}, indent=2), flush=True)
    if args.audit_only:
        return
    for kind in ("images", "labels", "gt_masks"):
        for split in SPLITS:
            (dst / kind / split).mkdir(parents=True, exist_ok=True)
    exported = []
    for i, record in enumerate(records):
        exported.append(export_case(record, dst, args.slice_stride))
        if (i + 1) % 10 == 0 or i == 0 or i + 1 == len(records):
            print(f"Prepared {i+1}/{len(records)} cases", flush=True)
    manifest.update(prepared=True, cases=exported)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    config = dict(path=str(dst), train="images/train", val="images/val", test="images/test", names={0: "whole_lesion"})
    (dst / "dataset_seg.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Ready: {dst / 'dataset_seg.yaml'}", flush=True)


if __name__ == "__main__":
    main()
