#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_files(folder: Path, exts: set[str]) -> dict[str, Path]:
    result = {}

    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue

        if p.suffix.lower() not in exts:
            continue

        if p.stem in result:
            raise RuntimeError(
                f"Duplicate stem '{p.stem}' in {folder}"
            )

        result[p.stem] = p

    return result


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert Kvasir mask to uint8 {0, 1}.

    Kvasir masks in this release are JPEG-like binary masks:
    background values near 0 and foreground near 255.
    """

    if mask is None:
        raise ValueError("mask is None")

    if mask.ndim != 2:
        raise ValueError(
            f"Expected grayscale mask, got {mask.shape}"
        )

    if int(mask.max()) <= 1:
        return (mask > 0).astype(np.uint8)

    _, binary255 = cv2.threshold(
        mask,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return (binary255 > 0).astype(np.uint8)


def contour_to_yolo_row(
    contour: np.ndarray,
    image_w: int,
    image_h: int,
) -> str | None:

    points = contour.reshape(-1, 2).astype(np.float32)

    if len(points) < 3:
        return None

    points[:, 0] /= float(image_w)
    points[:, 1] /= float(image_h)

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

    coords = " ".join(
        f"{v:.6f}"
        for v in points.reshape(-1)
    )

    return f"0 {coords}"


def direct_mask_polygons(
    binary01: np.ndarray,
    min_contour_area: float,
) -> list[str]:

    h, w = binary01.shape

    binary255 = (
        (binary01 > 0).astype(np.uint8)
        * 255
    )

    contours, _ = cv2.findContours(
        binary255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid = []

    for contour in contours:
        area = float(
            cv2.contourArea(contour)
        )

        if area < min_contour_area:
            continue

        row = contour_to_yolo_row(
            contour,
            w,
            h,
        )

        if row is not None:
            valid.append(
                (
                    area,
                    row,
                )
            )

    valid.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    return [
        row
        for _, row in valid
    ]


def bbox_guided_polygons(
    binary01: np.ndarray,
    boxes: list[dict],
    min_contour_area: float,
) -> list[str]:
    """
    Create exactly one mask polygon per official Kvasir bbox.

    Used only when whole-mask connected-component count does not
    match the number of annotated instances.
    """

    h, w = binary01.shape

    rows = []

    for box_idx, box in enumerate(boxes):
        xmin = int(
            max(
                0,
                min(
                    box["xmin"],
                    box["xmax"],
                ),
            )
        )

        ymin = int(
            max(
                0,
                min(
                    box["ymin"],
                    box["ymax"],
                ),
            )
        )

        xmax = int(
            min(
                w,
                max(
                    box["xmin"],
                    box["xmax"],
                ),
            )
        )

        ymax = int(
            min(
                h,
                max(
                    box["ymin"],
                    box["ymax"],
                ),
            )
        )

        if xmax <= xmin or ymax <= ymin:
            raise RuntimeError(
                f"Degenerate bbox #{box_idx}: {box}"
            )

        crop = binary01[
            ymin:ymax,
            xmin:xmax
        ]

        crop255 = (
            (crop > 0).astype(np.uint8)
            * 255
        )

        contours, _ = cv2.findContours(
            crop255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        candidates = []

        for contour in contours:
            area = float(
                cv2.contourArea(contour)
            )

            if area < min_contour_area:
                continue

            candidates.append(
                (
                    area,
                    contour,
                )
            )

        if not candidates:
            raise RuntimeError(
                "No valid foreground contour inside "
                f"bbox #{box_idx}: {box}"
            )

        # Largest foreground component within this official bbox.
        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        contour = candidates[0][1].copy()

        # Convert crop coordinates back to full-image coordinates.
        contour[:, 0, 0] += xmin
        contour[:, 0, 1] += ymin

        row = contour_to_yolo_row(
            contour,
            w,
            h,
        )

        if row is None:
            raise RuntimeError(
                f"Invalid polygon for bbox #{box_idx}"
            )

        rows.append(row)

    return rows


def split_ids(
    ids: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:

    ids = sorted(ids)

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_val = max(
        1,
        int(
            round(
                len(ids)
                * val_fraction
            )
        ),
    )

    return (
        sorted(ids[n_val:]),
        sorted(ids[:n_val]),
    )


def prepare_split(
    split: str,
    ids: list[str],
    images: dict[str, Path],
    masks: dict[str, Path],
    bbox_data: dict,
    dst: Path,
    min_contour_area: float,
) -> dict:

    image_out = (
        dst / "images" / split
    )

    label_out = (
        dst / "labels" / split
    )

    gt_out = (
        dst / "gt_masks" / split
    )

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
        "direct": 0,
        "bbox_guided": 0,
    }

    for stem in ids:
        src_image = images[stem]
        src_mask = masks[stem]

        shutil.copy2(
            src_image,
            image_out / src_image.name,
        )

        mask = cv2.imread(
            str(src_mask),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"Could not read {src_mask}"
            )

        binary01 = binarize_mask(
            mask
        )

        if int(binary01.sum()) == 0:
            stats["empty_masks"] += 1

        cv2.imwrite(
            str(
                gt_out
                / f"{stem}.png"
            ),
            binary01 * 255,
        )

        entry = bbox_data.get(stem)

        if entry is None:
            raise RuntimeError(
                f"No bbox annotation for {stem}"
            )

        boxes = entry.get(
            "bbox",
            [],
        )

        if not boxes:
            raise RuntimeError(
                f"No bbox instances for {stem}"
            )

        direct_rows = direct_mask_polygons(
            binary01,
            min_contour_area,
        )

        if len(direct_rows) == len(boxes):
            rows = direct_rows
            stats["direct"] += 1

        else:
            rows = bbox_guided_polygons(
                binary01,
                boxes,
                min_contour_area,
            )

            stats["bbox_guided"] += 1

        if len(rows) != len(boxes):
            raise RuntimeError(
                f"{stem}: "
                f"bbox={len(boxes)}, "
                f"polygon={len(rows)}"
            )

        (
            label_out
            / f"{stem}.txt"
        ).write_text(
            "\n".join(rows)
            + "\n",
            encoding="utf-8",
        )

        stats["images"] += 1
        stats["instances"] += len(rows)

    return stats


def write_yaml(
    dst: Path,
) -> Path:

    yaml_path = (
        dst / "dataset_seg.yaml"
    )

    payload = {
        "path": str(
            dst.resolve()
        ),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "polyp",
        },
    }

    yaml_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()

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

    images = collect_files(
        src / "images",
        IMAGE_EXTS,
    )

    masks = collect_files(
        src / "masks",
        MASK_EXTS,
    )

    bbox_path = (
        src / "kavsir_bboxes.json"
    )

    if not bbox_path.exists():
        raise FileNotFoundError(
            bbox_path
        )

    with open(
        bbox_path,
        "r",
        encoding="utf-8",
    ) as f:
        bbox_data = json.load(f)

    if set(images) != set(masks):
        raise RuntimeError(
            "Image/mask stems mismatch"
        )

    ids = sorted(images)

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{dst} exists. "
                "Use --overwrite."
            )

        shutil.rmtree(dst)

    train_ids, val_ids = split_ids(
        ids,
        args.val_fraction,
        args.seed,
    )

    print()
    print(
        "# KVASIR-SEG -> YOLO26-SEG"
    )

    print()
    print(
        "paired       :",
        len(ids),
    )

    train_stats = prepare_split(
        "train",
        train_ids,
        images,
        masks,
        bbox_data,
        dst,
        args.min_contour_area,
    )

    val_stats = prepare_split(
        "val",
        val_ids,
        images,
        masks,
        bbox_data,
        dst,
        args.min_contour_area,
    )

    (
        dst / "split_train.txt"
    ).write_text(
        "\n".join(train_ids) + "\n",
        encoding="utf-8",
    )

    (
        dst / "split_val.txt"
    ).write_text(
        "\n".join(val_ids) + "\n",
        encoding="utf-8",
    )

    yaml_path = write_yaml(
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
        train_stats["empty_masks"] != 0
        or val_stats["empty_masks"] != 0
    ):
        raise RuntimeError(
            "Empty masks detected"
        )

    print(
        "[PASS] dataset prepared"
    )


if __name__ == "__main__":
    main()
