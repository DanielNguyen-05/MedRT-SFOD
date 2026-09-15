#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolo26s-seg.pt')
    ap.add_argument('--data', default='dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='0')
    ap.add_argument('--seed', type=int, default=29)
    ap.add_argument('--optimizer', default='auto')
    ap.add_argument(
        '--project',
        default=str(Path(__file__).resolve().parents[3] / 'runs' / 'seg' / 'source')
    )
    ap.add_argument('--name', default='kvasir_yolo26s_seg')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f'Dataset YAML not found: {args.data}')

    epochs = 3 if args.smoke else args.epochs
    name = args.name + ('_smoke' if args.smoke else '')
    model = YOLO(args.weights, task='segment')

    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        seed=args.seed,
        deterministic=True,
        project=args.project,
        name=name,
        exist_ok=True,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,
        flipud=0.5,
        degrees=10.0,
        translate=0.05,
        scale=0.10,
        plots=True,
        verbose=True,
    )


if __name__ == '__main__':
    main()
