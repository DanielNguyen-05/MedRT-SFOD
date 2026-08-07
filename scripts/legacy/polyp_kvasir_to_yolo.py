"""Convert raw Kvasir-SEG layout into a standard Ultralytics YOLO dataset.

Raw Kvasir-SEG layout (as distributed by the dataset authors):
    <root>/images/*.jpg
    <root>/masks/*.jpg (or .png)
    <root>/kavsir_bboxes.json      # NOTE: "kavsir" is the dataset authors'
                                    # own filename typo, not ours — kept as-is
                                    # for compatibility with the official release.

`kavsir_bboxes.json` maps each image_id to a **list** of boxes:
    {"<image_id>": {"height": H, "width": W,
                     "bbox": [{"xmin":.., "ymin":.., "xmax":.., "ymax":..}, ...]}}

A prior version of this converter (`polyp_dataset_utils.py`, now removed) only
read `bbox[0]`, silently dropping every additional polyp in multi-polyp
images. This script iterates over the *entire* list.

Output layout (standard Ultralytics format, directly usable by
`YOLO(cfg).train(data=<dataset>.yaml)` with no extra path configuration):
    <out>/images/train/*.jpg
    <out>/images/val/*.jpg
    <out>/labels/train/*.txt   # contents depend on --task, see below
    <out>/labels/val/*.txt
    <out>/dataset_detect.yaml  # written when --task detect or both
    <out>/dataset_seg.yaml     # written when --task segment or both

--task detect  : labels/ = YOLO boxes  "class cx cy w h"            (normalized)
--task segment : labels/ = YOLO polygons "class x1 y1 x2 y2 ..."    (normalized)
--task both    : labels/ = boxes, labels_seg/ = polygons (see printed note —
                 dataset_seg.yaml will need labels_seg/ renamed to labels/ or
                 a second --dst run with --task segment, since Ultralytics
                 resolves labels from images/ by path substitution and does
                 not know about labels_seg/ on its own)

Segmentation polygons are extracted from the binary mask with
`cv2.findContours` (one polygon per external contour, so multi-polyp masks
with disconnected regions correctly produce multiple instances instead of
one merged blob).

Usage:
    python polyp_kvasir_to_yolo.py --src /path/to/Kvasir-SEG --dst /path/to/datasets/polyp_detect --task detect
    python polyp_kvasir_to_yolo.py --src /path/to/Kvasir-SEG --dst /path/to/datasets/polyp_seg --task segment
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

CLASS_NAMES = {0: "polyp"}


def find_mask_path(mask_dir: Path, image_id: str) -> Optional[Path]:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = mask_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def boxes_to_yolo_lines(boxes: List[dict], img_w: int, img_h: int) -> List[str]:
    """Convert every box in the list (not just the first) to a YOLO detection line."""
    lines = []
    for box in boxes:
        xmin, ymin, xmax, ymax = box["xmin"], box["ymin"], box["xmax"], box["ymax"]
        xmin, xmax = max(0, min(xmin, xmax)), min(img_w, max(xmin, xmax))
        ymin, ymax = max(0, min(ymin, ymax)), min(img_h, max(ymin, ymax))
        cx = ((xmin + xmax) / 2.0) / img_w
        cy = ((ymin + ymax) / 2.0) / img_h
        w = (xmax - xmin) / img_w
        h = (ymax - ymin) / img_h
        if w <= 0 or h <= 0:
            continue  # degenerate box, skip rather than write a zero-area label
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def mask_to_yolo_seg_lines(mask: np.ndarray, min_contour_area: float = 15.0) -> List[str]:
    """Extract one polygon per external contour in the binary mask.

    Each disconnected blob becomes its own instance line, so a mask with two
    separate polyps produces two label lines instead of one polygon that
    silently unions them.
    """
    h, w = mask.shape[:2]
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        contour = contour.reshape(-1, 2).astype(np.float32)
        contour[:, 0] /= w
        contour[:, 1] /= h
        coords = " ".join(f"{v:.6f}" for v in contour.flatten())
        lines.append(f"0 {coords}")
    return lines


def split_ids(image_ids: List[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_fraction)))
    return ids[n_val:], ids[:n_val]  # train, val


def write_dataset_yaml(path: Path, dataset_root: Path) -> None:
    content = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }
    with open(path, "w") as f:
        yaml.safe_dump(content, f, sort_keys=False)


def convert(src: Path, dst: Path, val_fraction: float, seed: int, task: str) -> None:
    image_dir = src / "images"
    mask_dir = src / "masks"
    bbox_path = src / "kavsir_bboxes.json"  # dataset authors' own filename, kept verbatim

    if not bbox_path.exists():
        raise FileNotFoundError(f"{bbox_path} not found — is --src pointing at the Kvasir-SEG root?")
    with open(bbox_path, "r", encoding="utf-8") as f:
        bbox_data = json.load(f)

    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    image_ids = [p.stem for p in image_paths]
    if not image_ids:
        raise FileNotFoundError(f"No images found under {image_dir}")

    train_ids, val_ids = split_ids(image_ids, val_fraction, seed)
    id_to_path = {p.stem: p for p in image_paths}

    write_seg_primary = task == "segment"  # polygons go straight into labels/
    write_seg_extra = task == "both"  # polygons go into a second labels_seg/ tree
    need_mask = write_seg_primary or write_seg_extra

    (dst / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "val").mkdir(parents=True, exist_ok=True)
    if write_seg_extra:
        (dst / "labels_seg" / "train").mkdir(parents=True, exist_ok=True)
        (dst / "labels_seg" / "val").mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "skipped_no_box": 0, "skipped_no_mask": 0}

    for split, ids in (("train", train_ids), ("val", val_ids)):
        for image_id in ids:
            src_img = id_to_path[image_id]
            shutil.copy2(src_img, dst / "images" / split / src_img.name)

            entry = bbox_data.get(image_id)
            det_lines: List[str] = []
            if entry is not None and entry.get("bbox"):
                img_w, img_h = entry.get("width"), entry.get("height")
                if img_w is None or img_h is None:
                    img = cv2.imread(str(src_img))
                    img_h, img_w = img.shape[:2]
                det_lines = boxes_to_yolo_lines(entry["bbox"], img_w, img_h)
            else:
                stats["skipped_no_box"] += 1

            seg_lines: List[str] = []
            if need_mask:
                mask_path = find_mask_path(mask_dir, image_id)
                if mask_path is None:
                    stats["skipped_no_mask"] += 1
                else:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    seg_lines = mask_to_yolo_seg_lines(mask) if mask is not None else []

            primary_lines = seg_lines if write_seg_primary else det_lines
            label_path = dst / "labels" / split / f"{image_id}.txt"
            label_path.write_text("\n".join(primary_lines) + ("\n" if primary_lines else ""))

            if write_seg_extra:
                seg_label_path = dst / "labels_seg" / split / f"{image_id}.txt"
                seg_label_path.write_text("\n".join(seg_lines) + ("\n" if seg_lines else ""))

            stats[split] += 1

    if task == "detect":
        write_dataset_yaml(dst / "dataset_detect.yaml", dst)
        print(f"[polyp_kvasir_to_yolo] wrote {dst / 'dataset_detect.yaml'} (labels/ = boxes)")
    elif task == "segment":
        write_dataset_yaml(dst / "dataset_seg.yaml", dst)
        print(f"[polyp_kvasir_to_yolo] wrote {dst / 'dataset_seg.yaml'} (labels/ = polygons)")
    else:  # both
        write_dataset_yaml(dst / "dataset_detect.yaml", dst)
        write_dataset_yaml(dst / "dataset_seg.yaml", dst)
        print(f"[polyp_kvasir_to_yolo] wrote {dst / 'dataset_detect.yaml'} (labels/ = boxes)")
        print(f"[polyp_kvasir_to_yolo] wrote {dst / 'dataset_seg.yaml'} (polygons are in labels_seg/, see note)")
        print(
            "[polyp_kvasir_to_yolo] NOTE (--task both only): dataset_seg.yaml still points "
            "train/val at images/, and Ultralytics resolves labels from images/ by path "
            "substitution — so training with dataset_seg.yaml as-is will read labels/ (the "
            "BOX labels), not labels_seg/. For segmentation training from a --task both "
            "export, either rename labels_seg/ to labels/ (and labels/ to labels_detect/ "
            "first if you still need the boxes), or simply re-run this converter with "
            "--task segment into a different --dst — cheap, since only the small .txt label "
            "files differ between runs."
        )

    print(
        f"[polyp_kvasir_to_yolo] train={stats['train']} val={stats['val']} "
        f"skipped_no_box={stats['skipped_no_box']} skipped_no_mask={stats['skipped_no_mask']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw Kvasir-SEG to standard Ultralytics YOLO format")
    parser.add_argument("--src", type=str, required=True, help="Path to raw Kvasir-SEG root (images/, masks/, kavsir_bboxes.json)")
    parser.add_argument("--dst", type=str, required=True, help="Output dataset root")
    parser.add_argument("--task", type=str, choices=["detect", "segment", "both"], default="detect")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(Path(args.src), Path(args.dst), args.val_fraction, args.seed, args.task)
