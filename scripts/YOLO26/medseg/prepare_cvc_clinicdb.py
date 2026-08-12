#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


def natural_key(p: Path):
    return int(p.stem) if p.stem.isdigit() else p.stem


def binarize_mask(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Convert grayscale CVC-ClinicDB GT mask to uint8 {0,1}.
    Uses Otsu because masks contain anti-aliased/intermediate grayscale values.
    """
    if mask is None:
        raise ValueError("mask is None")

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D grayscale mask, got {mask.shape}")

    if int(mask.max()) <= 1:
        return (mask > 0).astype(np.uint8), 0.0

    threshold, binary255 = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return (binary255 > 0).astype(np.uint8), float(threshold)


def mask_to_yolo_polygons(
    binary01: np.ndarray,
    min_contour_area: float = 15.0,
) -> list[str]:
    h, w = binary01.shape

    binary255 = binary01.astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        binary255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid = []

    for contour in contours:
        area = float(cv2.contourArea(contour))

        if area < min_contour_area:
            continue

        pts = contour.reshape(-1, 2).astype(np.float32)

        if len(pts) < 3:
            continue

        pts[:, 0] /= float(w)
        pts[:, 1] /= float(h)

        pts = np.clip(pts, 0.0, 1.0)

        coords = " ".join(
            f"{v:.6f}"
            for v in pts.reshape(-1)
        )

        valid.append((area, f"0 {coords}"))

    valid.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [row for _, row in valid]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--src",
        default="dataset/CVC-ClinicDB/PNG",
    )

    ap.add_argument(
        "--dst",
        default="dataset/CVC-ClinicDB-YOLO26",
    )

    ap.add_argument(
        "--min-contour-area",
        type=float,
        default=15.0,
    )

    ap.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    img_dir = src / "Original"
    mask_dir = src / "Ground Truth"

    if not img_dir.is_dir():
        raise FileNotFoundError(img_dir)

    if not mask_dir.is_dir():
        raise FileNotFoundError(mask_dir)

    images = {
        p.stem: p
        for p in img_dir.glob("*.png")
    }

    masks = {
        p.stem: p
        for p in mask_dir.glob("*.png")
    }

    if set(images) != set(masks):
        raise RuntimeError(
            "Original / Ground Truth pairing mismatch"
        )

    stems = sorted(
        images,
        key=lambda x: int(x) if x.isdigit() else x,
    )

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{dst} exists; use --overwrite"
            )

        shutil.rmtree(dst)

    image_out = dst / "images" / "target"
    label_out = dst / "labels" / "target"
    gt_out = dst / "gt_masks" / "target"

    image_out.mkdir(parents=True)
    label_out.mkdir(parents=True)
    gt_out.mkdir(parents=True)

    instances = []
    thresholds = []
    empty_masks = 0

    print("=" * 70)
    print("CVC-CLINICDB -> YOLO26-SEG")
    print("=" * 70)
    print("paired:", len(stems))

    for stem in stems:
        image_path = images[stem]
        mask_path = masks[stem]

        image = cv2.imread(str(image_path))

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {image_path}"
            )

        if mask is None:
            raise RuntimeError(
                f"Cannot read mask: {mask_path}"
            )

        if image.shape[:2] != mask.shape:
            raise RuntimeError(
                f"{stem}: image={image.shape[:2]} "
                f"mask={mask.shape}"
            )

        binary01, threshold = binarize_mask(mask)

        thresholds.append(threshold)

        if int(binary01.sum()) == 0:
            empty_masks += 1

        rows = mask_to_yolo_polygons(
            binary01,
            min_contour_area=args.min_contour_area,
        )

        if not rows:
            raise RuntimeError(
                f"{stem}: no valid polygon after binarization"
            )

        instances.append(len(rows))

        # Keep the original PNG image.
        shutil.copy2(
            image_path,
            image_out / f"{stem}.png",
        )

        # Binary GT mask used only by evaluator.
        ok = cv2.imwrite(
            str(gt_out / f"{stem}.png"),
            binary01 * 255,
        )

        if not ok:
            raise RuntimeError(
                f"Could not save GT mask for {stem}"
            )

        (label_out / f"{stem}.txt").write_text(
            "\n".join(rows) + "\n",
            encoding="utf-8",
        )

    # Ultralytics dataset definition.
    yaml_payload = {
        "path": str(dst.resolve()),
        # Same target images are listed for train/val here only so
        # Ultralytics can resolve the dataset structure.
        # SFDA adaptation code must NOT load target GT labels.
        "train": "images/target",
        "val": "images/target",
        "names": {
            0: "polyp",
        },
    }

    yaml_path = dst / "dataset_seg.yaml"

    yaml_path.write_text(
        yaml.safe_dump(
            yaml_payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    hist = Counter(instances)

    print()
    print("images       :", len(stems))
    print("empty masks  :", empty_masks)
    print("instances    :", sum(instances))

    print()
    print("Instance histogram:")

    for n in sorted(hist):
        print(
            f"  {n} instance(s): {hist[n]} images"
        )

    thresholds_np = np.asarray(
        thresholds,
        dtype=np.float32,
    )

    print()
    print(
        "Otsu threshold:",
        f"min={thresholds_np.min():.1f}",
        f"mean={thresholds_np.mean():.2f}",
        f"max={thresholds_np.max():.1f}",
    )

    print()
    print("dataset yaml :", yaml_path)

    if empty_masks != 0:
        raise RuntimeError(
            f"{empty_masks} empty masks detected"
        )

    print("[PASS] CVC-ClinicDB target prepared")


if __name__ == "__main__":
    main()
