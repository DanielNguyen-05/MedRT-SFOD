#!/usr/bin/env python3
"""How much 3D Dice is lost to the confidence threshold and to cross-class overwriting?

The label-fidelity check showed the polygon labels are not the bottleneck (polygon Dice
0.94-0.998 per class; ~0.84-0.96 after YOLO's prototype-mask resolution), yet the source
model reaches only NETC 0.009 / ET 0.145 / SNFH 0.54 / RC 0.42 on source_val. Two cheap
suspects remain, both on the evaluation side:

  1. The fixed confidence 0.25. Per-slice bbox debugging showed in-domain confidences around
     0.01-0.09, so correct instances may simply be thresholded away.
  2. Cross-class overwriting. brats2024_eval.regions_from_result() paints instances in
     ascending confidence, so a filled ET disk that scores higher than the NETC inside it
     erases NETC. 55-65% of NETC polygon pixels overlap another class in the labels.

One forward pass per slice at the lowest threshold, then every threshold and two overlap
policies are scored from the same predictions:

  semantic  - the official rule: one label per voxel, higher confidence wins.
  per_class - each class is the union of its own instances; classes may overlap. This is an
              upper bound on what overwriting costs, not a deployable output.

Composite regions TC = NETC+ET and WT = NETC+SNFH+ET are scored too. If TC is good while NETC
and ET are poor, the model finds the tumour core but confuses the two sub-regions.

Metrics are case-level 3D Dice with the official empty policy (both empty = 1), plus the mean
over GT-present cases, the number that is not inflated by empty-vs-empty matches. Surface
distances are skipped for speed. At threshold 0.25 the `semantic` row must reproduce
evaluation/source_val.json; the script prints that comparison as a sanity check.

Choose a threshold only on source_val (labelled source data, allowed by the protocol).
--role target_test is for post-hoc analysis only and is never a legitimate selection set.

  python scripts/YOLO26/medseg/brats2024_conf_sweep.py --direction t1n_to_t2w --stage source_only
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from brats2024_data import NAMES  # noqa: E402  (sibling module, same directory)

DIRECTIONS = ('t1n_to_t2w', 't2w_to_t1n', 't1c_to_t2f', 't2f_to_t1c')
STAGE_CHECKPOINT = {'source_only': Path('source/weights/best.pt'), 'adabn': Path('adabn/adabn.pt'),
                    'medrt_sfseg': Path('adapt/student_final.pt')}
RAW = {name: cls + 1 for cls, name in NAMES.items()}
REGIONS = {**{name: (raw,) for name, raw in RAW.items()},
           'TC': (RAW['NETC'], RAW['ET']), 'WT': (RAW['NETC'], RAW['SNFH'], RAW['ET'])}
POLICIES = ('semantic', 'per_class')


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def slice_counts(masks, classes, conf, gt, thresholds):
    """Return {(policy, threshold, region): (tp, fp, fn)} for one slice.

    masks: (N, H, W) bool, classes: (N,) int in 0..3, conf: (N,), gt: (H, W) raw labels.
    """
    out = {}
    gt_regions = {region: np.isin(gt, raws) for region, raws in REGIONS.items()}
    for t in thresholds:
        keep = np.flatnonzero(conf >= t)
        # semantic: official tie-break, ascending (conf, -index), later paints win
        semantic = np.zeros(gt.shape, np.uint8)
        for i in sorted(keep, key=lambda i: (conf[i], -i)):
            semantic[masks[i]] = classes[i] + 1
        per_class = {raw: np.zeros(gt.shape, bool) for raw in RAW.values()}
        for i in keep:
            per_class[int(classes[i]) + 1] |= masks[i]
        for region, raws in REGIONS.items():
            g = gt_regions[region]
            preds = {'semantic': np.isin(semantic, raws),
                     'per_class': np.logical_or.reduce([per_class[r] for r in raws])}
            for policy, pr in preds.items():
                out[(policy, t, region)] = (int((pr & g).sum()), int((pr & ~g).sum()), int((~pr & g).sum()))
    return out


def case_dice(tp, fp, fn):
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=REPO / 'configs/experiments/brats2024_sfseg.yaml')
    p.add_argument('--direction', required=True, choices=DIRECTIONS)
    p.add_argument('--stage', default='source_only', choices=tuple(STAGE_CHECKPOINT))
    p.add_argument('--role', default='source_val', choices=('source_val', 'target_test'))
    p.add_argument('--thresholds', type=float, nargs='+', default=[0.01, 0.02, 0.05, 0.1, 0.15, 0.25, 0.35, 0.5])
    p.add_argument('--max-cases', type=int, default=0, help='0 = every case of the role.')
    p.add_argument('--weights', help='Explicit checkpoint path, overrides --stage.')
    p.add_argument('--device', default=None)
    p.add_argument('--batch', type=int, default=None)
    p.add_argument('--out-dir', type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = args.device if args.device is not None else cfg.get('device', '0')
    batch = args.batch or cfg['batch']
    imgsz = cfg['imgsz']
    official = cfg['evaluation']['confidence']
    thresholds = sorted(set(args.thresholds) | {official})
    dst = resolve_path(cfg['prepared_root']) / 'full'
    run = resolve_path(cfg['project']) / 'full' / args.direction
    checkpoint = resolve_path(args.weights) if args.weights else run / STAGE_CHECKPOINT[args.stage]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    source, target = args.direction.split('_to_')
    modality = source if args.role == 'source_val' else target
    if args.role == 'target_test':
        print('[note] target_test sweep is post-hoc analysis only; never pick a threshold from it.', flush=True)
    out_dir = args.out_dir or run / '_conf_sweep' / f'{args.stage}_{args.role}'
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((dst / 'manifest.json').read_text())
    records = [c for c in manifest['cases'] if c['role'] == args.role]
    if args.max_cases:
        records = records[:args.max_cases]

    from ultralytics import YOLO
    model = YOLO(str(checkpoint), task='segment')

    # per (policy, threshold, region): list of per-case (dice, gt_present)
    per_case = defaultdict(list)
    t0 = time.time()
    for ci, case in enumerate(records):
        cid = case['case_id']
        with np.load(dst / 'evaluation_masks' / args.role / f'{cid}.npz') as data:
            gt_volume = data['mask']
        shape = (case['shape'][1], case['shape'][0])
        if gt_volume.shape[:2] != shape:
            raise ValueError(f'GT geometry mismatch for {cid}: {gt_volume.shape[:2]} != {shape}')
        totals = defaultdict(lambda: [0, 0, 0])
        paths = [str(dst / modality / 'images' / args.role / f'{cid}_z{z:03d}.png') for z in case['slices']]
        for start in range(0, len(paths), batch):
            chunk = paths[start:start + batch]
            results = model.predict(source=chunk, imgsz=imgsz, batch=batch, device=device, conf=min(thresholds),
                                    retina_masks=True, verbose=False, stream=True, max_det=300)
            for offset, result in enumerate(results):
                gt = gt_volume[:, :, start + offset]
                if result.masks is None or result.boxes is None or len(result.boxes) == 0:
                    masks = np.zeros((0, *shape), bool)
                    classes = np.zeros(0, int)
                    conf = np.zeros(0)
                else:
                    masks = result.masks.data.detach().cpu().numpy() > .5
                    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
                    conf = result.boxes.conf.detach().cpu().numpy()
                    if masks.shape[1:] != shape:
                        raise ValueError(f'Prediction geometry mismatch: {masks.shape[1:]} != {shape}')
                if not gt.any() and len(conf) == 0:
                    continue
                for key, (tp, fp, fn) in slice_counts(masks, classes, conf, gt, thresholds).items():
                    acc = totals[key]
                    acc[0] += tp; acc[1] += fp; acc[2] += fn
        gt_present = {region: bool(np.isin(gt_volume, raws).any()) for region, raws in REGIONS.items()}
        for policy in POLICIES:
            for t in thresholds:
                for region in REGIONS:
                    tp, fp, fn = totals[(policy, t, region)]
                    per_case[(policy, t, region)].append((case_dice(tp, fp, fn), gt_present[region]))
        if (ci + 1) % 10 == 0 or ci + 1 == len(records):
            print(f'{args.role}: {ci + 1}/{len(records)} cases, {time.time() - t0:.0f}s', flush=True)

    summary = {}
    for (policy, t, region), rows in per_case.items():
        dice = [d for d, _ in rows]
        present = [d for d, g in rows if g]
        summary.setdefault(policy, {}).setdefault(region, {})[t] = dict(
            dice=float(np.mean(dice)), dice_gt_present=float(np.mean(present)) if present else None,
            gt_present_cases=len(present))

    def table(policy, key):
        head = '| Region | ' + ' | '.join(f'{t:g}' + (' (official)' if t == official else '') for t in thresholds) + ' | best |'
        lines = [head, '|---|' + '---:|' * (len(thresholds) + 1)]
        for region in REGIONS:
            vals = {t: summary[policy][region][t][key] for t in thresholds}
            valid = {t: v for t, v in vals.items() if v is not None}
            best = max(valid, key=valid.get) if valid and len(set(valid.values())) > 1 else None
            cells = ' | '.join('—' if vals[t] is None else (f'**{vals[t]:.3f}**' if t == best else f'{vals[t]:.3f}')
                               for t in thresholds)
            lines.append(f'| {region} | {cells} | {"—" if best is None else f"{best:g}"} |')
        return lines

    lines = [f'# Confidence sweep — {args.direction}, {args.stage}, {args.role} ({len(records)} cases)', '',
             f'Checkpoint: `{checkpoint}`', '']
    for policy in POLICIES:
        for key, label in (('dice_gt_present', 'Dice on GT-present cases (not inflated by empty-vs-empty)'),
                           ('dice', 'Dice, official empty policy (both empty = 1)')):
            lines += [f'## {policy} — {label}', ''] + table(policy, key) + ['']

    official_json = run / 'evaluation' / ('source_val.json' if args.role == 'source_val' else f'{args.stage}_target_test.json')
    if official_json.is_file() and not args.max_cases and not args.weights:
        ref = json.loads(official_json.read_text())['summary']
        checks = []
        for name in RAW:
            mine = summary['semantic'][name][official]['dice']
            theirs = ref[name]['dice']['case_mean']
            checks.append(f'{name} {mine:.3f} vs {theirs:.3f}' + ('' if abs(mine - theirs) < 0.005 else ' MISMATCH'))
        lines += ['## Sanity check against the official evaluation', '',
                  f'`semantic` at {official:g} vs `{official_json.name}`: ' + '; '.join(checks), '']

    report = '\n'.join(lines) + '\n'
    (out_dir / 'conf_sweep.md').write_text(report)
    (out_dir / 'conf_sweep.json').write_text(json.dumps(dict(
        direction=args.direction, stage=args.stage, role=args.role, checkpoint=str(checkpoint),
        thresholds=thresholds, official=official,
        summary={p: {r: {str(t): v for t, v in d.items()} for r, d in rs.items()} for p, rs in summary.items()}),
        indent=2) + '\n')
    print('\n' + report)
    print(f'Saved {out_dir / "conf_sweep.md"}')


if __name__ == '__main__':
    main()
