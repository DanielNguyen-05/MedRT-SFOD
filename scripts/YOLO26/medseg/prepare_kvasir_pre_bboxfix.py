#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MASK_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def collect_files(folder: Path, exts: tuple[str, ...]) -> dict[str, Path]:
    files = {}

    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            if p.stem in files:
                raise RuntimeError(
                    f"Duplicate stem '{p.stem}' found in {folder}"
                )

            files[p.stem] = p

    return files


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert a Kvasir-SEG mask to uint8 binary values {0, 1}.

    Supports:
      - masks encoded as 0/1
      - masks encoded as grayscale / 0..255
      - JPEG/grayscale compression artifacts
    """

    if mask is None:
        raise ValueError("Mask is None")

    if mask.ndim != 2:
        raise ValueError(
            f"Expected grayscale 2D mask, got shape={mask.shape}"
        )

    mask = mask.astype(np.uint8)

    min_value = int(mask.min())
    max_value = int(mask.max())

    # Literal binary 0/1 mask.
    if max_value <= 1:
        return (mask > 0).astype(np.uint8)

    # Constant mask should never occur for Kvasir foreground annotations.
    if min_value == max_value:
        return (mask > 0).astype(np.uint8)

    # Robust binarization for normal 0..255 masks and compressed masks.
    _, binary255 = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return (binary255 > 0).astype(np.uint8)


def binary_mask_to_yolo_polygons(
    binary01: np.ndarray,
    min_contour_area: float = 15.0,
) -> list[str]:
    """
    Convert binary polyp regions to YOLO instance-segmentation polygons.

    Each disconnected external foreground region becomes one instance.
    """

    h, w = binary01.shape

    binary255 = (
        (binary01 > 0).astype(np.uint8) * 255
    )

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

        points = contour.reshape(-1, 2).astype(np.float32)

        if len(points) < 3:
            continue

        # Normalize to [0, 1].
        points[:, 0] /= float(w)
        points[:, 1] /= float(h)

        points[:, 0] = np.clip(
            points[:, 0],
            0.0,
            1.0,
        )

        points[:, 1] = np.clip(
            points[:, 1],
            0.0,
            1.0,
        )

        valid.append(
            (
                area,
                points,
            )
        )

    # Larger foreground regions first for deterministic labels.
    valid.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    rows = []

    for _, points in valid:
        coords = " ".join(
            f"{value:.6f}"
            for value in points.reshape(-1)
        )

        rows.append(
            f"0 {coords}"
        )

    return rows


def split_ids(
    ids: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:

    if not 0.0 < val_fraction < 1.0:
        raise ValueError(
            "--val-fraction must be between 0 and 1"
        )

    ids = sorted(ids)

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_val = max(
        1,
        int(round(len(ids) * val_fraction)),
    )

    val_ids = sorted(
        ids[:n_val]
    )

    train_ids = sorted(
        ids[n_val:]
    )

    return train_ids, val_ids


def prepare_split(
    split: str,
    ids: list[str],
    images: dict[str, Path],
    masks: dict[str, Path],
    dst: Path,
    min_contour_area: float,
) -> dict[str, int]:

    image_out = dst / "images" / split
    label_out = dst / "labels" / split
    gt_out = dst / "gt_masks" / split

    image_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    gt_out.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats = {
        "images": 0,
        "instances": 0,
        "empty_masks": 0,
    }

    for stem in ids:
        image_path = images[stem]
        mask_path = masks[stem]

        # Keep original image extension.
        shutil.copy2(
            image_path,
            image_out / image_path.name,
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"Cannot read mask: {mask_path}"
            )

        binary01 = binarize_mask(mask)

        if binary01.sum() == 0:
            stats["empty_masks"] += 1

        # Preserve binary GT mask for Dice / IoU evaluation.
        cv2.imwrite(
            str(
                gt_out /
                f"{stem}.png"
            ),
            binary01 * 255,
        )

        rows = binary_mask_to_yolo_polygons(
            binary01,
            min_contour_area=min_contour_area,
        )

        label_path = (
            label_out /
            f"{stem}.txt"
        )

        label_path.write_text(
            "\n".join(rows)
            + ("\n" if rows else ""),
            encoding="utf-8",
        )

        stats["images"] += 1
        stats["instances"] += len(rows)

    return stats


def write_dataset_yaml(
    dst: Path,
) -> Path:

    path = dst / "dataset_seg.yaml"

    data = {
        "path": str(
            dst.resolve()
        ),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "polyp",
        },
    }

    path.write_text(
        yaml.safe_dump(
            data,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Kvasir-SEG for "
            "YOLO26 instance segmentation"
        )
    )

    parser.add_argument(
        "--src",
        default="dataset/Kvasir-SEG",
    )

    parser.add_argument(
        "--dst",
        default="dataset/Kvasir-SEG-YOLO26",
    )

    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=29,
    )

    parser.add_argument(
        "--min-contour-area",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    image_dir = src / "images"
    mask_dir = src / "masks"

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Missing: {image_dir}"
        )

    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Missing: {mask_dir}"
        )

    images = collect_files(
        image_dir,
        IMAGE_EXTS,
    )

    masks = collect_files(
        mask_dir,
        MASK_EXTS,
    )

    image_ids = set(images)
    mask_ids = set(masks)

    if image_ids != mask_ids:
        print(
            "image only:",
            len(
                image_ids - mask_ids
            ),
        )

        print(
            "mask only:",
            len(
                mask_ids - image_ids
            ),
        )

        raise RuntimeError(
            "Image/mask pairing mismatch"
        )

    ids = sorted(image_ids)

    print(
        "# KVASIR-SEG -> YOLO26-SEG"
    )

    print()
    print(
        "paired       :",
        len(ids),
    )

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{dst} already exists; "
                "use --overwrite"
            )

        shutil.rmtree(dst)

    train_ids, val_ids = split_ids(
        ids,
        args.val_fraction,
        args.seed,
    )

    train_stats = prepare_split(
        "train",
        train_ids,
        images,
        masks,
        dst,
        args.min_contour_area,
    )

    val_stats = prepare_split(
        "val",
        val_ids,
        images,
        masks,
        dst,
        args.min_contour_area,
    )

    (dst / "split_train.txt").write_text(
        "\n".join(train_ids) + "\n",
        encoding="utf-8",
    )

    (dst / "split_val.txt").write_text(
        "\n".join(val_ids) + "\n",
        encoding="utf-8",
    )

    yaml_path = write_dataset_yaml(
        dst
    )

    print(
        "train        :",
        len(train_ids),
        train_stats,
    )

    print(
        "val          :",
        len(val_ids),
        val_stats,
    )

    print(
        "seed         :",
        args.seed,
    )

    print(
        "dataset yaml :",
        yaml_path,
    )

    if (
        train_stats["empty_masks"] > 0
        or val_stats["empty_masks"] > 0
    ):
        raise RuntimeError(
            "Empty masks detected. "
            "Dataset preparation is NOT valid."
        )

    if (
        train_stats["instances"] == 0
        or val_stats["instances"] == 0
    ):
        raise RuntimeError(
            "No segmentation instances created."
        )

    print(
        "[PASS] dataset prepared"
    )


if __name__ == "__main__":
    main()
