#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import torch
import ultralytics
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolo26s-seg.pt')
    ap.add_argument('--data', default='dataset/Kvasir-SEG-YOLO26/dataset_seg.yaml')
    args = ap.parse_args()

    yolo = YOLO(args.weights, task='segment')
    model = yolo.model
    head = model.model[-1]
    params = sum(p.numel() for p in model.parameters())

    print('=' * 72)
    print('YOLO26-S-SEG PREFLIGHT')
    print('=' * 72)
    print('ultralytics :', ultralytics.__version__)
    print('local path  :', ultralytics.__file__)
    print('torch       :', torch.__version__)
    print('CUDA        :', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU         :', torch.cuda.get_device_name(0))
    print('dataset     :', Path(args.data), Path(args.data).exists())
    print('model       :', model.__class__.__name__)
    print('head        :', head.__class__.__name__)
    print('end2end     :', getattr(model, 'end2end', None))
    print('nc          :', getattr(head, 'nc', None))
    print('nm          :', getattr(head, 'nm', None))
    print('npr         :', getattr(head, 'npr', None))
    print('params      :', params)
    print('params M    :', params / 1e6)

    if 'segment' not in head.__class__.__name__.lower():
        raise RuntimeError(f'Expected segmentation head, got {head.__class__.__name__}')
    if not getattr(model, 'end2end', False):
        raise RuntimeError('Expected end2end=True')
    print('[PASS] YOLO26-S-Seg ready')


if __name__ == '__main__':
    main()
