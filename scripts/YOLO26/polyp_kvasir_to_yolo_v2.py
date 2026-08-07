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
--task both    : writes TWO independent valid datasets:
                 <out>/detect/ and <out>/segment/, sharing the exact same split

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


def _convert_single(
    src: Path,
    dst: Path,
    train_ids: List[str],
    val_ids: List[str],
    task: str,
    bbox_data: dict,
    id_to_path: dict[str, Path],
) -> dict:
    if task not in {"detect", "segment"}:
        raise ValueError(task)
    mask_dir = src / "masks"
    (dst / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dst / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dst / "labels" / "val").mkdir(parents=True, exist_ok=True)
    stats = {"train": 0, "val": 0, "empty_labels": 0, "missing_mask": 0}

    for split, ids in (("train", train_ids), ("val", val_ids)):
        for image_id in ids:
            src_img = id_to_path[image_id]
            shutil.copy2(src_img, dst / "images" / split / src_img.name)
            lines: List[str] = []
            if task == "detect":
                entry = bbox_data.get(image_id)
                if entry is not None and entry.get("bbox"):
                    img_w, img_h = entry.get("width"), entry.get("height")
                    if img_w is None or img_h is None:
                        img = cv2.imread(str(src_img)); img_h, img_w = img.shape[:2]
                    lines = boxes_to_yolo_lines(entry["bbox"], img_w, img_h)
            else:
                mask_path = find_mask_path(mask_dir, image_id)
                if mask_path is None:
                    stats["missing_mask"] += 1
                else:
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        lines = mask_to_yolo_seg_lines(mask)
            if not lines:
                stats["empty_labels"] += 1
            (dst / "labels" / split / f"{image_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            stats[split] += 1

    yaml_name = "dataset_detect.yaml" if task == "detect" else "dataset_seg.yaml"
    write_dataset_yaml(dst / yaml_name, dst)
    return stats


def convert(src: Path, dst: Path, val_fraction: float, seed: int, task: str) -> None:
    image_dir = src / "images"
    bbox_path = src / "kavsir_bboxes.json"
    if not bbox_path.exists():
        raise FileNotFoundError(f"{bbox_path} not found — is --src pointing at the Kvasir-SEG root?")
    bbox_data = json.loads(bbox_path.read_text(encoding="utf-8"))
    image_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_dir}")
    ids = [p.stem for p in image_paths]
    train_ids, val_ids = split_ids(ids, val_fraction, seed)
    id_to_path = {p.stem: p for p in image_paths}

    # Persist the split so detection/segmentation and future ablations use the
    # exact same images.  For the final paper, cross-dataset SOURCE->TARGET
    # adaptation should be defined at dataset level; this intra-Kvasir split is
    # only the source-supervised train/val split.
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "split_manifest.json").write_text(json.dumps({"seed": seed, "train": train_ids, "val": val_ids}, indent=2))

    if task == "both":
        det_stats = _convert_single(src, dst / "detect", train_ids, val_ids, "detect", bbox_data, id_to_path)
        seg_stats = _convert_single(src, dst / "segment", train_ids, val_ids, "segment", bbox_data, id_to_path)
        print(f"[Kvasir v2] detection dataset: {dst/'detect'}  stats={det_stats}")
        print(f"[Kvasir v2] segmentation dataset: {dst/'segment'}  stats={seg_stats}")
        print("[Kvasir v2] Both YAMLs are directly usable; no labels/ renaming is needed.")
    else:
        stats = _convert_single(src, dst, train_ids, val_ids, task, bbox_data, id_to_path)
        print(f"[Kvasir v2] {task} dataset: {dst} stats={stats}")

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
