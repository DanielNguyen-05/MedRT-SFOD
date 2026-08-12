#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

EPS = 1e-8
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def div(a: float, b: float) -> float:
    return float(a / (b + EPS))


def counts_and_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {
        'dice': div(2 * tp, 2 * tp + fp + fn),
        'iou': div(tp, tp + fp + fn),
        'precision': div(tp, tp + fp),
        'sensitivity': div(tp, tp + fn),
        'specificity': div(tn, tn + fp),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


def union_masks(result, h: int, w: int) -> np.ndarray:
    if result.masks is None or result.masks.data is None or result.masks.data.numel() == 0:
        return np.zeros((h, w), dtype=np.uint8)
    masks = result.masks.data.detach().float()
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    merged = masks.amax(dim=0, keepdim=True).unsqueeze(0)
    if tuple(merged.shape[-2:]) != (h, w):
        merged = F.interpolate(merged, size=(h, w), mode='nearest')
    return (merged[0, 0] > 0.5).cpu().numpy().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='runs/seg/source/kvasir_yolo26s_seg/weights/best.pt')
    ap.add_argument('--data', default='dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml')
    ap.add_argument('--images', default='dataset/Kvasir-SEG-YOLO26/images/val')
    ap.add_argument('--gt-masks', default='dataset/Kvasir-SEG-YOLO26/gt_masks/val')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--device', default='0')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--out', default='runs/seg/benchmarks/kvasir_yolo26s_seg_source_metrics.json')
    args = ap.parse_args()

    model_path = Path(args.model)
    image_dir = Path(args.images)
    gt_dir = Path(args.gt_masks)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not image_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError('Validation images or GT masks directory missing')

    model = YOLO(str(model_path), task='segment')
    native = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        plots=False,
        verbose=True,
    )

    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    rows = []
    totals = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

    for i, image_path in enumerate(image_paths, 1):
        gt_path = gt_dir / f'{image_path.stem}.png'
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            raise FileNotFoundError(gt_path)
        gt = (gt > 0).astype(np.uint8)

        result = model.predict(
            source=str(image_path), imgsz=args.imgsz, conf=args.conf,
            device=args.device, verbose=False,
        )[0]
        pred = union_masks(result, *gt.shape)
        m = counts_and_metrics(pred, gt)
        m['image'] = image_path.name
        rows.append(m)
        for k in totals:
            totals[k] += int(m[k])
        if i == 1 or i % 25 == 0 or i == len(image_paths):
            print(f'[{i:04d}/{len(image_paths):04d}] {image_path.name}')

    keys = ['dice', 'iou', 'precision', 'sensitivity', 'specificity']
    macro = {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}
    tp, fp, fn, tn = totals['tp'], totals['fp'], totals['fn'], totals['tn']
    global_metrics = {
        'dice': div(2 * tp, 2 * tp + fp + fn),
        'iou': div(tp, tp + fp + fn),
        'precision': div(tp, tp + fp),
        'sensitivity': div(tp, tp + fn),
        'specificity': div(tn, tn + fp),
        **totals,
    }

    params = sum(p.numel() for p in model.model.parameters())
    payload = {
        'model': str(model_path),
        'num_images': len(image_paths),
        'imgsz': args.imgsz,
        'conf': args.conf,
        'params': int(params),
        'params_M': params / 1e6,
        'native': {
            'box_map50': float(native.box.map50),
            'box_map50_95': float(native.box.map),
            'mask_map50': float(native.seg.map50),
            'mask_map50_95': float(native.seg.map),
            'speed': native.speed,
        },
        'medical_macro': macro,
        'medical_global': global_metrics,
        'per_image': rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    print('=' * 72)
    print('SOURCE SEGMENTATION RESULTS')
    print('=' * 72)
    print(f"Mask mAP50       : {payload['native']['mask_map50']:.6f}")
    print(f"Mask mAP50-95    : {payload['native']['mask_map50_95']:.6f}")
    print(f"Macro Dice       : {macro['dice']:.6f}")
    print(f"Macro IoU        : {macro['iou']:.6f}")
    print(f"Macro Precision  : {macro['precision']:.6f}")
    print(f"Macro Sensitivity: {macro['sensitivity']:.6f}")
    print(f"Macro Specificity: {macro['specificity']:.6f}")
    print(f"Params           : {params/1e6:.3f} M")
    print('Saved            :', out)


if __name__ == '__main__':
    main()
