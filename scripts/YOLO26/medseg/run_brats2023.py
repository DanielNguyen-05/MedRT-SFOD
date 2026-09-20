#!/usr/bin/env python3
"""Train the repository's YOLO26-S-Seg baseline; evaluate held-out BraTS cases.

This is a supervised binary baseline, not a source-free adaptation experiment or
the official multiclass, lesion-wise BraTS challenge evaluator.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO


def scores(tp: int, fp: int, fn: int) -> dict:
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        "precision": tp / (tp + fp) if tp + fp else (0.0 if fn else 1.0),
        "recall": tp / (tp + fn) if tp + fn else (0.0 if fp else 1.0),
    }


def summarize(rows: list[dict]) -> dict:
    # Average repeated scans within each patient before averaging patients.
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient_id"]].append(row)
    values = {key: np.array([np.mean([r[key] for r in visits]) for visits in patients.values()])
              for key in ("dice", "iou", "precision", "recall")}
    rng = np.random.default_rng(29)
    dice = values["dice"]
    boot = [float(rng.choice(dice, len(dice), replace=True).mean()) for _ in range(1000)]
    return dict(cases=len(rows), patients=len(patients),
                empty_ground_truth_cases=sum(r.get("tp", 0) + r.get("fn", 0) == 0 for r in rows),
                patient_macro={k: float(v.mean()) for k, v in values.items()},
                dice_patient_bootstrap_95ci=list(map(float, np.percentile(boot, [2.5, 97.5]))) )


def evaluate(model, root: Path, manifest: dict, split: str, args) -> dict:
    rows = []
    for case in manifest["cases"]:
        if case["split"] != split:
            continue
        cid = case["case_id"]
        images = [str(root / "images" / split / f"{cid}_z{z:03d}.png") for z in case["slices"]]
        tp = fp = fn = negative_slices = false_positive_slices = 0
        # Small batches bound memory; retina_masks removes letterbox padding correctly.
        for start in range(0, len(images), args.batch):
            batch_paths = images[start:start+args.batch]
            results = model.predict(source=batch_paths, imgsz=args.imgsz,
                                    batch=args.batch, device=args.device, conf=args.conf,
                                    retina_masks=True, verbose=False, stream=True)
            # Some local Ultralytics list loaders rename inputs to image0.jpg.
            # The prediction iterator preserves input order; use original paths.
            for image_path, result in zip(batch_paths, results, strict=True):
                gt_file = root / "gt_masks" / split / Path(image_path).name
                gt = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    raise FileNotFoundError(gt_file)
                gt = gt > 0
                pred = np.zeros_like(gt)
                if result.masks is not None:
                    pred = (result.masks.data > 0.5).any(dim=0).cpu().numpy()
                    if pred.shape != gt.shape:
                        raise ValueError(f"Unaligned prediction at {result.path}: {pred.shape} != {gt.shape}")
                tp += int((pred & gt).sum())
                fp += int((pred & ~gt).sum())
                fn += int((~pred & gt).sum())
                if not gt.any():
                    negative_slices += 1
                    false_positive_slices += bool(pred.any())
        rows.append(dict(case_id=cid, patient_id=case["patient_id"], domain=case["domain"],
                         slices=len(images), tp=tp, fp=fp, fn=fn, **scores(tp, fp, fn),
                         negative_slices=negative_slices, false_positive_slices=int(false_positive_slices)))
        if len(rows) % 20 == 0:
            print(f"{split}: evaluated {len(rows)} cases", flush=True)
    if not rows:
        raise ValueError(f"Empty {split} split")
    return dict(overall=summarize(rows),
                by_domain={d: summarize([r for r in rows if r["domain"] == d]) for d in sorted({r["domain"] for r in rows})},
                cases=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("dataset/BraTS2023-YOLO26/dataset_seg.yaml"))
    ap.add_argument("--weights", default="yolo26s-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--conf", type=float, default=0.25, help="Fixed before testing; never tuned on test")
    ap.add_argument("--project", type=Path, default=REPO / "runs" / "seg" / "brats2023")
    ap.add_argument("--name", default="yolo26s_wt")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="Resume last.pt in the same project/name")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch < 1 or not 0 < args.conf < 1:
        ap.error("epochs/batch must be positive; confidence must be between 0 and 1")
    if args.resume and args.eval_only:
        ap.error("--resume and --eval-only are mutually exclusive")
    root = args.data.resolve().parent
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest["prepared"]:
        raise ValueError("Manifest is audit-only/incomplete; prepare the dataset first")
    # Resolve the dataset relative to its current location after copying to Colab/GPU.
    run_dir = args.project.resolve() / args.name
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Use a fresh --name to preserve existing results: {run_dir}")
    if args.resume and not (run_dir / "weights" / "last.pt").is_file():
        raise FileNotFoundError(run_dir / "weights" / "last.pt")
    config = yaml.safe_load(args.data.read_text())
    config["path"] = str(root)
    for split in ("train", "val", "test"):
        patients = {c["patient_id"] for c in manifest["cases"] if c["split"] == split}
        others = {c["patient_id"] for c in manifest["cases"] if c["split"] != split}
        if not patients or patients & others:
            raise ValueError(f"Empty split or patient leakage in {split}")
        expected = {f"{c['case_id']}_z{z:03d}.png" for c in manifest["cases"] if c["split"] == split for z in c["slices"]}
        if expected != {p.name for p in (root / "images" / split).glob("*.png")}:
            raise ValueError(f"Image/manifest mismatch in {split}")
        config[split] = str(root / "images" / split)
    protocol_file = run_dir / "protocol.json"
    if args.resume:
        original = json.loads(protocol_file.read_text())
        if original["split_sha256"] != manifest["split_sha256"] or original["data_root"] != str(root):
            raise ValueError("Resume requires the original dataset/split at the same location")
    model = YOLO(str(run_dir / "weights" / "last.pt") if args.resume else args.weights, task="segment")
    run_dir.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume:
        protocol_file.write_text(json.dumps(dict(split_sha256=manifest["split_sha256"], data_root=str(root)), indent=2))
    run_yaml = run_dir / "dataset_resolved.yaml"
    run_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    started = time.monotonic()
    checkpoint = args.weights
    if not args.eval_only:
        train_options = dict(data=str(run_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                    workers=args.workers, device=args.device, seed=args.seed, deterministic=True,
                    project=str(args.project.resolve()), name=args.name, exist_ok=True,
                    optimizer="AdamW", lr0=0.001, patience=20,
                    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0,
                    mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
                    fliplr=0.5, flipud=0.0, degrees=10.0, translate=0.05, scale=0.10,
                    amp=args.device not in ("cpu", "mps"), plots=True)
        if args.resume:
            model.train(resume=True, device=args.device, workers=args.workers)
        else:
            model.train(**train_options)
        checkpoint = str(run_dir / "weights" / "best.pt")
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        model = YOLO(checkpoint, task="segment")
    training_seconds = time.monotonic() - started
    report = dict(method="Supervised YOLO26 binary whole-lesion baseline",
                  checkpoint=str(checkpoint), split_sha256=manifest["split_sha256"],
                  smoke_subset=manifest["smoke_subset"],
                  overlap_unit="full 3D case voxel counts" if manifest["slice_stride"] == 1 else "subsampled slices (smoke only)",
                  not_official_brats_challenge_score=True,
                  checkpoint_selection="validation box + mask mAP50-95 fitness (Ultralytics best.pt)" if not args.eval_only else "supplied checkpoint; training provenance not verified",
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  torch_version=torch.__version__, training_seconds=training_seconds,
                  empty_mask_convention="Dice/IoU=1 if both empty, 0 if only one empty; inspect empty_ground_truth_cases, especially for smoke subsets")
    for split in ("val", "test"):
        metrics = model.val(data=str(run_yaml), split=split, imgsz=args.imgsz, batch=args.batch,
                            workers=args.workers, device=args.device, conf=0.001,
                            project=str(run_dir), name=f"{split}_official_yolo", plots=False, verbose=False)
        report[split] = evaluate(model, root, manifest, split, args)
        report[split]["yolo_metrics"] = {k: float(v) for k, v in metrics.results_dict.items()}
        report[split]["confidence_for_overlap"] = args.conf
        (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"{split}: {json.dumps(report[split]['overall'])}", flush=True)
    report["total_seconds"] = time.monotonic() - started
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Results: {run_dir / 'evaluation.json'}", flush=True)


if __name__ == "__main__":
    main()
