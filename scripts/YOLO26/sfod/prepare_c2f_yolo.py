#!/usr/bin/env python3
"""
Prepare Cityscapes -> Foggy Cityscapes (C2F) object-detection data in Ultralytics YOLO format.

Protocol used:
- Source: clear Cityscapes train/val.
- Target: Foggy Cityscapes beta=0.02 only.
- Detection classes (8):
    0 person
    1 rider
    2 car
    3 truck
    4 bus
    5 train
    6 motorcycle
    7 bicycle
- Bounding boxes are derived from gtFine polygon JSONs.
- Crowd/group labels such as "persongroup" and "cargroup" are ignored.
- Images are symlinked by default to avoid duplicating the dataset.

Expected raw layout:
  dataset/
    gtFine_trainvaltest/gtFine/{train,val}/<city>/*_gtFine_polygons.json
    leftImg8bit_trainvaltest/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
    leftImg8bit_trainvaltest_foggy/leftImg8bit_foggy/{train,val}/<city>/*_leftImg8bit_foggy_beta_0.02.png

Output:
  dataset/c2f_yolo/
    cityscapes/
      images/{train,val}/...
      labels/{train,val}/...
      cityscapes.yaml
    foggy_cityscapes/
      images/{train,val}/...
      labels/{train,val}/...
      foggy_cityscapes.yaml

Note:
Target labels are written only so that final validation is possible.
Your Stage 0/1 and Stage 2 source-free adaptation scripts should consume target images
without using these labels for optimization/model selection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml

CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("dataset"))
    p.add_argument("--out", type=Path, default=Path("dataset/c2f_yolo"))
    p.add_argument("--fog-beta", type=str, default="0.02")
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image files instead of symlinking them (uses much more disk space).",
    )
    return p.parse_args()


def safe_link_or_copy(src: Path, dst: Path, copy_images: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        # Relative symlink keeps the prepared dataset movable with the repo tree.
        rel = os.path.relpath(src.resolve(), start=dst.parent.resolve())
        dst.symlink_to(rel)


def polygon_to_box(polygon, img_w: int, img_h: int):
    if not polygon or len(polygon) < 3:
        return None
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    xmin = max(0.0, min(xs))
    ymin = max(0.0, min(ys))
    xmax = min(float(img_w), max(xs))
    ymax = min(float(img_h), max(ys))
    if xmax <= xmin or ymax <= ymin:
        return None

    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return cx, cy, bw, bh


def labels_from_json(json_path: Path):
    data = json.loads(json_path.read_text())
    img_w = int(data["imgWidth"])
    img_h = int(data["imgHeight"])

    lines = []
    counts = Counter()
    ignored_group = 0

    for obj in data.get("objects", []):
        label = obj.get("label", "")

        # Cityscapes group/crowd annotations are not single-instance boxes.
        if label.endswith("group"):
            ignored_group += 1
            continue

        if label not in CLASS_TO_ID:
            continue

        box = polygon_to_box(obj.get("polygon", []), img_w, img_h)
        if box is None:
            continue

        cls_id = CLASS_TO_ID[label]
        cx, cy, bw, bh = box
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        counts[label] += 1

    return lines, counts, ignored_group


def clear_image_path(clear_root: Path, split: str, city: str, image_id: str) -> Path:
    return clear_root / split / city / f"{image_id}_leftImg8bit.png"


def foggy_image_path(foggy_root: Path, split: str, city: str, image_id: str, beta: str) -> Path:
    return foggy_root / split / city / f"{image_id}_leftImg8bit_foggy_beta_{beta}.png"


def process_split(
    split: str,
    gt_root: Path,
    clear_root: Path,
    foggy_root: Path,
    out_root: Path,
    beta: str,
    copy_images: bool,
):
    clear_img_out = out_root / "cityscapes" / "images" / split
    clear_lbl_out = out_root / "cityscapes" / "labels" / split
    fog_img_out = out_root / "foggy_cityscapes" / "images" / split
    fog_lbl_out = out_root / "foggy_cityscapes" / "labels" / split

    total_images = 0
    total_boxes = 0
    class_counts = Counter()
    ignored_groups = 0
    missing_clear = []
    missing_foggy = []

    json_paths = sorted((gt_root / split).glob("*/*_gtFine_polygons.json"))
    if not json_paths:
        raise RuntimeError(f"No gtFine polygon JSON files found under {gt_root / split}")

    for json_path in json_paths:
        city = json_path.parent.name
        image_id = json_path.name.replace("_gtFine_polygons.json", "")

        clear_src = clear_image_path(clear_root, split, city, image_id)
        fog_src = foggy_image_path(foggy_root, split, city, image_id, beta)

        if not clear_src.exists():
            missing_clear.append(str(clear_src))
            continue
        if not fog_src.exists():
            missing_foggy.append(str(fog_src))
            continue

        lines, counts, ignored = labels_from_json(json_path)

        # Flat output names are safe because Cityscapes image IDs include the city name.
        clear_dst = clear_img_out / clear_src.name
        fog_dst = fog_img_out / fog_src.name
        clear_lbl = clear_lbl_out / f"{image_id}_leftImg8bit.txt"
        fog_lbl = fog_lbl_out / f"{image_id}_leftImg8bit_foggy_beta_{beta}.txt"

        safe_link_or_copy(clear_src, clear_dst, copy_images)
        safe_link_or_copy(fog_src, fog_dst, copy_images)

        clear_lbl.parent.mkdir(parents=True, exist_ok=True)
        fog_lbl.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(lines)
        if text:
            text += "\n"
        clear_lbl.write_text(text)
        fog_lbl.write_text(text)

        total_images += 1
        total_boxes += len(lines)
        class_counts.update(counts)
        ignored_groups += ignored

    if missing_clear or missing_foggy:
        print(f"[WARN] split={split}: missing clear={len(missing_clear)}, missing foggy={len(missing_foggy)}")
        if missing_clear[:3]:
            print("  example missing clear:", *missing_clear[:3], sep="\n    ")
        if missing_foggy[:3]:
            print("  example missing foggy:", *missing_foggy[:3], sep="\n    ")

    print(f"[{split}] images={total_images} boxes={total_boxes} ignored_group_objects={ignored_groups}")
    print(f"[{split}] class_counts={dict(class_counts)}")
    return total_images


def write_yaml(path: Path, root: Path):
    content = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(CLASSES)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(content, sort_keys=False))


def main():
    args = parse_args()
    ds = args.dataset_root.resolve()
    out = args.out.resolve()

    gt_root = ds / "gtFine_trainvaltest" / "gtFine"
    clear_root = ds / "leftImg8bit_trainvaltest" / "leftImg8bit"
    foggy_root = ds / "leftImg8bit_trainvaltest_foggy" / "leftImg8bit_foggy"

    for p in (gt_root, clear_root, foggy_root):
        if not p.exists():
            raise FileNotFoundError(p)

    print("Preparing C2F YOLO dataset")
    print("  clear:", clear_root)
    print("  foggy:", foggy_root)
    print("  gt:", gt_root)
    print("  beta:", args.fog_beta)
    print("  output:", out)
    print("  image mode:", "copy" if args.copy_images else "symlink")

    counts = {}
    for split in ("train", "val"):
        counts[split] = process_split(
            split=split,
            gt_root=gt_root,
            clear_root=clear_root,
            foggy_root=foggy_root,
            out_root=out,
            beta=args.fog_beta,
            copy_images=args.copy_images,
        )

    write_yaml(out / "cityscapes" / "cityscapes.yaml", out / "cityscapes")
    write_yaml(out / "foggy_cityscapes" / "foggy_cityscapes.yaml", out / "foggy_cityscapes")

    print("\n[OK] conversion complete")
    print("  clear yaml:", out / "cityscapes" / "cityscapes.yaml")
    print("  foggy yaml:", out / "foggy_cityscapes" / "foggy_cityscapes.yaml")
    print(f"  train images={counts['train']} val images={counts['val']}")

    # Strong sanity checks for the official fine split.
    if counts["train"] != 2975 or counts["val"] != 500:
        print(
            "[WARN] Expected standard Cityscapes fine split counts 2975 train / 500 val. "
            f"Observed {counts['train']} / {counts['val']}."
        )


if __name__ == "__main__":
    main()
