#!/usr/bin/env python3
"""
Prepare reverse MedRT-SFSeg direction:

    SOURCE: CVC-ClinicDB (labeled source-supervised training)
    TARGET: Kvasir-SEG   (unlabeled during AdaBN / Stage-2)

This creates NEW prepared directories, so the existing Kvasir->CVC experiment
is not overwritten.

Defaults
--------
dataset/CVC-ClinicDB-YOLO26-SOURCE/
    images/train, images/val
    labels/train, labels/val
    gt_masks/train, gt_masks/val
    split_train.txt, split_val.txt
    dataset_seg.yaml

dataset/Kvasir-SEG-YOLO26-TARGET/
    images/target
    labels/target        # evaluation only
    gt_masks/target      # evaluation only
    dataset_seg.yaml

Protocol
--------
- CVC source split: deterministic 80/20, seed=29 by default.
- Kvasir target: ALL images are used as the transductive unlabeled target set.
- Target labels/GT exist only for post-training evaluation. AdaBN/Stage-2 must
  receive only images/target and never labels/target or gt_masks/target.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        raise RuntimeError(f"No images found in: {folder}")
    return paths


def find_by_stem(folder: Path, stem: str) -> Path:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"):
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    matches = [p for p in folder.iterdir() if p.is_file() and p.stem == stem]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Cannot find mask for stem={stem!r} in {folder}")


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Use --overwrite to rebuild it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def put_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            dst.hardlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(mode)


def load_binary_mask(mask_path: Path, image_hw: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    h, w = image_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def mask_to_yolo_seg_lines(
    mask: np.ndarray,
    min_contour_area: float,
) -> list[str]:
    """Extract one YOLO segmentation polygon per external connected component."""
    h, w = mask.shape
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    lines: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            continue

        coords: list[str] = []
        for x, y in pts:
            xn = float(np.clip(x / max(w, 1), 0.0, 1.0))
            yn = float(np.clip(y / max(h, 1), 0.0, 1.0))
            coords.extend([f"{xn:.6f}", f"{yn:.6f}"])
        lines.append("0 " + " ".join(coords))
    return lines


def write_example(
    image_path: Path,
    mask_path: Path,
    image_out: Path,
    label_out: Path,
    gt_out: Path,
    min_contour_area: float,
    link_mode: str,
) -> dict:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    h, w = bgr.shape[:2]

    mask = load_binary_mask(mask_path, (h, w))
    lines = mask_to_yolo_seg_lines(mask, min_contour_area)

    put_file(image_path, image_out, link_mode)

    label_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.write_text(
        ("\n".join(lines) + "\n") if lines else "",
        encoding="utf-8",
    )

    gt_out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(gt_out), mask * 255):
        raise RuntimeError(f"Failed to write {gt_out}")

    return {
        "image": image_path.name,
        "mask": mask_path.name,
        "height": int(h),
        "width": int(w),
        "instances": len(lines),
        "foreground_pixels": int(mask.sum()),
    }


def write_yaml(root: Path, train_rel: str, val_rel: str) -> Path:
    payload = {
        "path": str(root.resolve()),
        "train": train_rel,
        "val": val_rel,
        "names": {0: "polyp"},
        "nc": 1,
    }
    path = root / "dataset_seg.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def prepare_cvc_source(args) -> dict:
    raw = Path(args.cvc_raw).resolve()
    image_dir = raw / "Original"
    mask_dir = raw / "Ground Truth"
    dst = Path(args.cvc_source_out).resolve()

    images = image_files(image_dir)
    n = len(images)
    n_val = max(1, int(round(n * args.val_fraction)))
    n_val = min(n_val, n - 1)

    order = list(range(n))
    random.Random(args.seed).shuffle(order)
    val_idx = set(order[:n_val])

    train_images = [p for i, p in enumerate(images) if i not in val_idx]
    val_images = [p for i, p in enumerate(images) if i in val_idx]

    reset_dir(dst, args.overwrite)

    records: dict[str, list[dict]] = {"train": [], "val": []}
    for split, paths in (("train", train_images), ("val", val_images)):
        for image_path in paths:
            mask_path = find_by_stem(mask_dir, image_path.stem)
            rec = write_example(
                image_path=image_path,
                mask_path=mask_path,
                image_out=dst / "images" / split / image_path.name,
                label_out=dst / "labels" / split / f"{image_path.stem}.txt",
                gt_out=dst / "gt_masks" / split / f"{image_path.stem}.png",
                min_contour_area=args.min_contour_area,
                link_mode=args.link_mode,
            )
            records[split].append(rec)

    (dst / "split_train.txt").write_text(
        "\n".join(r["image"] for r in records["train"]) + "\n",
        encoding="utf-8",
    )
    (dst / "split_val.txt").write_text(
        "\n".join(r["image"] for r in records["val"]) + "\n",
        encoding="utf-8",
    )
    yaml_path = write_yaml(dst, "images/train", "images/val")

    return {
        "role": "source",
        "dataset": "CVC-ClinicDB",
        "raw": str(raw),
        "prepared": str(dst),
        "yaml": str(yaml_path),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "num_total": n,
        "num_train": len(train_images),
        "num_val": len(val_images),
    }


def prepare_kvasir_target(args) -> dict:
    raw = Path(args.kvasir_raw).resolve()
    image_dir = raw / "images"
    mask_dir = raw / "masks"
    dst = Path(args.kvasir_target_out).resolve()

    images = image_files(image_dir)
    reset_dir(dst, args.overwrite)

    records = []
    for image_path in images:
        mask_path = find_by_stem(mask_dir, image_path.stem)
        rec = write_example(
            image_path=image_path,
            mask_path=mask_path,
            image_out=dst / "images" / "target" / image_path.name,
            label_out=dst / "labels" / "target" / f"{image_path.stem}.txt",
            gt_out=dst / "gt_masks" / "target" / f"{image_path.stem}.png",
            min_contour_area=args.min_contour_area,
            link_mode=args.link_mode,
        )
        records.append(rec)

    # This YAML is primarily for evaluation. Adaptation gets only images/target.
    yaml_path = write_yaml(dst, "images/target", "images/target")

    return {
        "role": "target",
        "dataset": "Kvasir-SEG",
        "raw": str(raw),
        "prepared": str(dst),
        "yaml": str(yaml_path),
        "num_target": len(images),
        "adaptation_reads": str(dst / "images" / "target"),
        "evaluation_gt": str(dst / "gt_masks" / "target"),
        "target_labels_policy": "evaluation only; never passed to adaptation",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare CVC-ClinicDB -> Kvasir-SEG reverse SFDA direction."
    )
    ap.add_argument("--cvc-raw", default="dataset/CVC-ClinicDB/PNG")
    ap.add_argument("--kvasir-raw", default="dataset/Kvasir-SEG")
    ap.add_argument(
        "--cvc-source-out",
        default="dataset/CVC-ClinicDB-YOLO26-SOURCE",
    )
    ap.add_argument(
        "--kvasir-target-out",
        default="dataset/Kvasir-SEG-YOLO26-TARGET",
    )
    ap.add_argument("--val-fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--min-contour-area", type=float, default=15.0)
    ap.add_argument(
        "--link-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not (0.0 < args.val_fraction < 1.0):
        raise ValueError("--val-fraction must be between 0 and 1")

    cvc = prepare_cvc_source(args)
    kvasir = prepare_kvasir_target(args)

    manifest = {
        "direction": "CVC-ClinicDB -> Kvasir-SEG",
        "source_free_adaptation": True,
        "source": cvc,
        "target": kvasir,
    }
    manifest_path = (
        Path(args.kvasir_target_out).resolve().parent
        / "cvc_to_kvasir_reverse_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("CVC-ClinicDB -> Kvasir-SEG preparation complete")
    print("=" * 78)
    print(
        f"CVC source : total={cvc['num_total']} "
        f"train={cvc['num_train']} val={cvc['num_val']}"
    )
    print(f"CVC YAML   : {cvc['yaml']}")
    print(f"Kvasir tgt : {kvasir['num_target']} images")
    print(f"Kvasir YAML: {kvasir['yaml']}")
    print(f"Manifest   : {manifest_path}")
    print()
    print("[SOURCE-FREE CHECK]")
    print("AdaBN/Stage-2 must receive ONLY:")
    print(" ", kvasir["adaptation_reads"])
    print("Do NOT pass labels/target or gt_masks/target to adaptation.")
    print("[PASS]")


if __name__ == "__main__":
    main()
