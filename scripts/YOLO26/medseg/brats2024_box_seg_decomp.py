#!/usr/bin/env python3
"""Does a wrong detected box cause the wrong segmentation? Box-vs-mask error decomposition.

In a YOLO segmentation head every instance mask is cropped to its own predicted box, so a
missing, misplaced, too small or wrongly classed box removes that segmentation no matter how
good the prototype mask is. This script splits every ground-truth pixel of each class into:

  TP            covered by a predicted mask of the same class
  miss: no box  not inside any same-class predicted box. Detection/box failure: the mask
                could never have reached it.
     of which other-class box   inside a box of another class: class confusion, not localisation
  miss: in box  inside a same-class box but the mask left it out: mask failure

and reports, per class:
  - pooled Dice,
  - the box-limited ceiling: Dice if the masks were perfect inside the predicted same-class
    boxes and empty outside. If this ceiling is far below 1, the boxes are the bottleneck.
    If it is high while Dice is low, the masks are.
  - instance level: share of GT components (>= --min-area px) with a same-class box at
    IoU >= 0.5, mean best box IoU, and mean mask Dice of matched instances (mask quality
    once the box is right).

Masks are taken per class without cross-class overwriting, so this isolates box vs mask
effects. brats2024_conf_sweep.py already measures what the overwrite rule costs.

Choose thresholds on source_val. --role target_test is post-hoc analysis only.
--modality overrides the image folder, e.g. t2w_inv after brats2024_ablation.py inverted it.

  python scripts/YOLO26/medseg/brats2024_box_seg_decomp.py --direction t1n_to_t2w --stage source_only
  python scripts/YOLO26/medseg/brats2024_box_seg_decomp.py --direction t1n_to_t2w --stage medrt_sfseg --role target_test
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from brats2024_data import NAMES  # noqa: E402

DIRECTIONS = ('t1n_to_t2w', 't2w_to_t1n', 't1c_to_t2f', 't2f_to_t1c')
STAGE_CHECKPOINT = {'source_only': Path('source/weights/best.pt'), 'adabn': Path('adabn/adabn.pt'),
                    'medrt_sfseg': Path('adapt/student_final.pt')}


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def box_canvas(boxes, shape):
    canvas = np.zeros(shape, bool)
    h, w = shape
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        x2, y2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = True
    return canvas


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def slice_stats(masks, boxes, classes, conf, gt, thresholds, min_area, iou_match=0.5):
    """Per (threshold, class): pixel buckets and instance records for one slice."""
    pix, inst = {}, defaultdict(list)
    comps = {}
    for cls in NAMES:
        g = (gt == cls + 1).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
        comps[cls] = [(labels == k, (stats[k, 0], stats[k, 1], stats[k, 0] + stats[k, 2], stats[k, 1] + stats[k, 3]))
                      for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= min_area]
    for t in thresholds:
        alive = conf >= t
        for cls in NAMES:
            g = gt == cls + 1
            same = np.flatnonzero(alive & (classes == cls))
            other = np.flatnonzero(alive & (classes != cls))
            mask = np.logical_or.reduce(masks[same]) if len(same) else np.zeros(gt.shape, bool)
            in_box = box_canvas(boxes[same], gt.shape)
            in_other = box_canvas(boxes[other], gt.shape)
            miss = g & ~mask
            pix[(t, cls)] = dict(gt=int(g.sum()), tp=int((mask & g).sum()), fp=int((mask & ~g).sum()),
                                 miss_no_box=int((miss & ~in_box).sum()),
                                 miss_other_box=int((miss & ~in_box & in_other).sum()),
                                 miss_in_box=int((miss & in_box).sum()),
                                 gt_in_box=int((g & in_box).sum()))
            for comp_mask, comp_box in comps[cls]:
                best_iou, best = 0.0, None
                for i in same:
                    iou = box_iou(comp_box, boxes[i])
                    if iou > best_iou:
                        best_iou, best = iou, i
                rec = dict(box_iou=best_iou, matched=best_iou >= iou_match)
                if rec['matched']:
                    m = masks[best]
                    tp = int((m & comp_mask).sum())
                    rec['mask_dice'] = 2 * tp / (int(m.sum()) + int(comp_mask.sum()))
                inst[(t, cls)].append(rec)
    return pix, inst


def fmt(v, pct=False):
    if v is None:
        return '—'
    return f'{100 * v:.1f}%' if pct else f'{v:.3f}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=REPO / 'configs/experiments/brats2024_sfseg.yaml')
    p.add_argument('--direction', required=True, choices=DIRECTIONS)
    p.add_argument('--stage', default='source_only', choices=tuple(STAGE_CHECKPOINT))
    p.add_argument('--role', default='source_val', choices=('source_val', 'target_test'))
    p.add_argument('--modality', default=None, help='Image folder override, e.g. t2w_inv.')
    p.add_argument('--weights', default=None, help='Explicit checkpoint, overrides --stage.')
    p.add_argument('--thresholds', type=float, nargs='+', default=None,
                   help='Default: the official evaluation confidence and 0.05.')
    p.add_argument('--min-area', type=int, default=10, help='Ignore smaller GT components at instance level.')
    p.add_argument('--max-cases', type=int, default=0)
    p.add_argument('--device', default=None)
    p.add_argument('--out-dir', type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    official = cfg['evaluation']['confidence']
    thresholds = sorted(set(args.thresholds or [official, 0.05]))
    device = args.device if args.device is not None else cfg.get('device', '0')
    dst = resolve_path(cfg['prepared_root']) / 'full'
    run = resolve_path(cfg['project']) / 'full' / args.direction
    checkpoint = resolve_path(args.weights) if args.weights else run / STAGE_CHECKPOINT[args.stage]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source, target = args.direction.split('_to_')
    modality = args.modality or (source if args.role == 'source_val' else target)
    if args.role == 'target_test':
        print('[note] target_test decomposition is post-hoc analysis only.', flush=True)
    tag = f'{args.stage if not args.weights else Path(args.weights).stem}_{args.role}_{modality}'
    out_dir = args.out_dir or run / '_box_seg_decomp' / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((dst / 'manifest.json').read_text())
    records = [c for c in manifest['cases'] if c['role'] == args.role]
    if args.max_cases:
        records = records[:args.max_cases]

    from ultralytics import YOLO
    model = YOLO(str(checkpoint), task='segment')
    pix_tot = defaultdict(lambda: defaultdict(int))
    inst_all = defaultdict(list)
    t0 = time.time()
    for ci, case in enumerate(records):
        with np.load(dst / 'evaluation_masks' / args.role / f"{case['case_id']}.npz") as data:
            gt_volume = data['mask']
        shape = (case['shape'][1], case['shape'][0])
        paths = [str(dst / modality / 'images' / args.role / f"{case['case_id']}_z{z:03d}.png") for z in case['slices']]
        for start in range(0, len(paths), cfg['batch']):
            chunk = paths[start:start + cfg['batch']]
            results = model.predict(source=chunk, imgsz=cfg['imgsz'], batch=cfg['batch'], device=device,
                                    conf=min(thresholds), retina_masks=True, verbose=False, stream=True, max_det=300)
            for offset, result in enumerate(results):
                gt = gt_volume[:, :, start + offset]
                if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
                    if not gt.any():
                        continue
                    masks, boxes = np.zeros((0, *shape), bool), np.zeros((0, 4))
                    classes, conf = np.zeros(0, int), np.zeros(0)
                else:
                    masks = result.masks.data.detach().cpu().numpy() > .5
                    boxes = result.boxes.xyxy.detach().cpu().numpy()
                    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                    conf = result.boxes.conf.detach().cpu().numpy()
                    if masks.shape[1:] != shape:
                        raise ValueError(f'Prediction geometry mismatch: {masks.shape[1:]} != {shape}')
                pix, inst = slice_stats(masks, boxes, classes, conf, gt, thresholds, args.min_area)
                for key, d in pix.items():
                    for k, v in d.items():
                        pix_tot[key][k] += v
                for key, recs in inst.items():
                    inst_all[key].extend(recs)
        if (ci + 1) % 10 == 0 or ci + 1 == len(records):
            print(f'{args.role}: {ci + 1}/{len(records)} cases, {time.time() - t0:.0f}s', flush=True)

    lines = [f'# Box vs mask error decomposition — {args.direction}, {tag} ({len(records)} cases)', '',
             f'Checkpoint: `{checkpoint}`. Per-class masks without cross-class overwriting.', '']
    summary = {}
    for t in thresholds:
        lines += [f'## Confidence ≥ {t:g}' + (' (official)' if t == official else ''), '',
                  '| Class | Dice | Box-limited ceiling | GT px hit | Miss: no same-class box | …of which inside other-class box | Miss: inside box (mask) | FP / GT px |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for cls, name in NAMES.items():
            d = pix_tot[(t, cls)]
            g = d['gt']
            dice = 2 * d['tp'] / (2 * d['tp'] + d['fp'] + (g - d['tp'])) if g + d['fp'] else None
            ceiling = 2 * d['gt_in_box'] / (d['gt_in_box'] + g) if g else None
            row = dict(dice=dice, box_ceiling=ceiling, hit=d['tp'] / g if g else None,
                       miss_no_box=d['miss_no_box'] / g if g else None,
                       miss_other_box=d['miss_other_box'] / g if g else None,
                       miss_in_box=d['miss_in_box'] / g if g else None, fp_over_gt=d['fp'] / g if g else None)
            recs = inst_all[(t, cls)]
            matched = [r for r in recs if r['matched']]
            row.update(instances=len(recs), detected=len(matched) / len(recs) if recs else None,
                       mean_box_iou=float(np.mean([r['box_iou'] for r in recs])) if recs else None,
                       mask_dice_matched=float(np.mean([r['mask_dice'] for r in matched])) if matched else None)
            summary.setdefault(str(t), {})[name] = row
            lines.append(f"| {name} | {fmt(row['dice'])} | {fmt(row['box_ceiling'])} | {fmt(row['hit'], True)} | "
                         f"{fmt(row['miss_no_box'], True)} | {fmt(row['miss_other_box'], True)} | "
                         f"{fmt(row['miss_in_box'], True)} | {fmt(row['fp_over_gt'])} |")
        lines += ['', f'Instance level (GT components ≥ {args.min_area} px, same-class box IoU ≥ 0.5 = detected):', '',
                  '| Class | GT instances | Detected | Mean best box IoU | Mask Dice when detected |',
                  '|---|---:|---:|---:|---:|']
        for name in NAMES.values():
            r = summary[str(t)][name]
            lines.append(f"| {name} | {r['instances']} | {fmt(r['detected'], True)} | {fmt(r['mean_box_iou'])} | "
                         f"{fmt(r['mask_dice_matched'])} |")
        lines.append('')
    lines += ['How to read it: a low box-limited ceiling or a large "no same-class box" share means the',
              'detections are the bottleneck (missing/misplaced/mis-classed boxes cut the masks). A high ceiling',
              'with a large "inside box" share, or a low mask Dice on detected instances, means the masks are.', '']
    report = '\n'.join(lines)
    (out_dir / 'box_seg_decomp.md').write_text(report + '\n')
    (out_dir / 'box_seg_decomp.json').write_text(json.dumps(dict(
        direction=args.direction, stage=args.stage, role=args.role, modality=modality, checkpoint=str(checkpoint),
        thresholds=thresholds, min_area=args.min_area, summary=summary), indent=2) + '\n')
    print('\n' + report)
    print(f'Saved {out_dir / "box_seg_decomp.md"}')


if __name__ == '__main__':
    main()
