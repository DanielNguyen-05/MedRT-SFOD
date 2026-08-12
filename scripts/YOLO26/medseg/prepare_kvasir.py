#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MASK_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def collect(folder: Path, exts: set[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            if p.stem in out:
                raise RuntimeError(f'Duplicate stem {p.stem!r} in {folder}')
            out[p.stem] = p
    return out


def mask_to_rows(mask: np.ndarray, min_area: float) -> list[str]:
    h, w = mask.shape
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows: list[str] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            continue
        pts[:, 0] = np.clip(pts[:, 0] / w, 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0.0, 1.0)
        coords = ' '.join(f'{v:.6f}' for v in pts.reshape(-1))
        rows.append(f'0 {coords}')
    return rows


def write_split(split: str, ids: list[str], images: dict[str, Path], masks: dict[str, Path], dst: Path, min_area: float):
    image_out = dst / 'images' / split
    label_out = dst / 'labels' / split
    gt_out = dst / 'gt_masks' / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    gt_out.mkdir(parents=True, exist_ok=True)

    instances = 0
    empty = 0
    for stem in ids:
        src_img = images[stem]
        shutil.copy2(src_img, image_out / src_img.name)

        mask = cv2.imread(str(masks[stem]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f'Cannot read mask: {masks[stem]}')
        binary = (mask > 0).astype(np.uint8)
        cv2.imwrite(str(gt_out / f'{stem}.png'), binary * 255)

        rows = mask_to_rows(binary, min_area)
        (label_out / f'{stem}.txt').write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
        instances += len(rows)
        empty += int(not rows)
    return {'images': len(ids), 'instances': instances, 'empty_masks': empty}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='dataset/Kvasir-SEG')
    ap.add_argument('--dst', default='dataset/Kvasir-SEG-YOLO26')
    ap.add_argument('--val-fraction', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=29)
    ap.add_argument('--min-contour-area', type=float, default=15.0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    image_dir, mask_dir = src / 'images', src / 'masks'
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError('Expected dataset/Kvasir-SEG/images and dataset/Kvasir-SEG/masks')

    images = collect(image_dir, IMAGE_EXTS)
    masks = collect(mask_dir, MASK_EXTS)
    if set(images) != set(masks):
        raise RuntimeError(f'Image/mask mismatch: images={len(images)} masks={len(masks)} paired={len(set(images)&set(masks))}')

    ids = sorted(images)
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * args.val_fraction))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f'{dst} exists; use --overwrite to rebuild')
        shutil.rmtree(dst)

    train_stats = write_split('train', train_ids, images, masks, dst, args.min_contour_area)
    val_stats = write_split('val', val_ids, images, masks, dst, args.min_contour_area)

    (dst / 'split_train.txt').write_text('\n'.join(train_ids) + '\n', encoding='utf-8')
    (dst / 'split_val.txt').write_text('\n'.join(val_ids) + '\n', encoding='utf-8')
    yaml_path = dst / 'dataset_seg.yaml'
    yaml_path.write_text(yaml.safe_dump({
        'path': str(dst.resolve()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'polyp'},
    }, sort_keys=False), encoding='utf-8')

    print('=' * 72)
    print('KVASIR-SEG -> YOLO26-SEG')
    print('=' * 72)
    print('paired       :', len(ids))
    print('train        :', len(train_ids), train_stats)
    print('val          :', len(val_ids), val_stats)
    print('seed         :', args.seed)
    print('dataset yaml :', yaml_path)
    print('[PASS] dataset prepared')


if __name__ == '__main__':
    main()
