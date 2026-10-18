#!/usr/bin/env python3
"""How faithful are the BraTS2024 YOLO polygon labels to the real masks?

The source model is weak even in-domain (t1n source_val: SNFH 0.54, ET 0.145, NETC 0.009
Dice on GT-present cases). Before retraining anything, this measures the ceiling the
*supervision itself* imposes. No model and no GPU are needed. It only reads files that
`brats2024_experiments.py --stage prepare` already wrote:

  <prepared>/full/<modality>/labels/<role>/<case>_zNNN.txt   (YOLO polygons, the training targets)
  <prepared>/full/evaluation_masks/<role>/<case>.npz         (exact raw-label masks)

Table A rasterizes the polygons the model was actually trained on and compares them with
the exact masks, per class. The converter writes one external contour per connected
component, so holes are filled (an ET rim becomes a disk that also covers NETC) and
degenerate contours (< 3 points or zero area: thin or 1-px structures) are dropped. The
table reports:
  - pooled and mean per-slice Dice of the training polygons vs the real class mask,
  - slices where the class exists but no polygon survived,
  - the share of polygon pixels that are not this class in the GT,
  - the share of polygon pixels also claimed by another class's polygon (conflicting
    supervision for an instance-segmentation head),
  - the same Dice after an approximate round trip through YOLO's prototype-mask
    resolution (imgsz / mask_ratio), which is what the seg head can actually express.

Table B builds labels from the GT under alternative rules, so you can see which
reformulation removes the ceiling before you spend GPU time:
  - `external`: the current converter rule, recomputed from GT. It should match Table A.
    If it does not, polygon orientation/scaling is wrong somewhere.
  - `holes`: outer contours minus their holes. A YOLO polygon can carry this with a
    zero-width bridge cut from each hole to the outer contour.
  - the composite regions TC = NETC+ET and WT = NETC+SNFH+ET, which are the standard
    BraTS evaluation regions and much closer to filled blobs.

Labels are identical for every modality (the converter writes the same rows for all
four), so --modality only selects which copy of the label folder to read.

Run where dataset/BraTS2024-SFSeg exists (the Z800):

  python scripts/YOLO26/medseg/brats2024_label_fidelity.py --role source_val
  python scripts/YOLO26/medseg/brats2024_label_fidelity.py --role source_train --max-cases 150
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
from brats2024_data import NAMES  # noqa: E402  (sibling module, same directory)

RAW = {name: cls + 1 for cls, name in NAMES.items()}          # NETC=1 SNFH=2 ET=3 RC=4
REGIONS = {**{name: (raw,) for name, raw in RAW.items()},
           'TC': (RAW['NETC'], RAW['ET']),
           'WT': (RAW['NETC'], RAW['SNFH'], RAW['ET'])}


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def keep_contour(contour):
    """Same rule as brats2024_data.polygons(): drop < 3 points or zero area."""
    return len(contour) >= 3 and cv2.contourArea(contour) > 0


def read_polygons(path, h, w):
    """YOLO seg txt -> {class_id: [int32 (N,2) point arrays in pixel coordinates]}."""
    out = defaultdict(list)
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        xy = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2) * [w, h]
        out[cls].append(np.rint(xy).astype(np.int32))
    return out


def raster(polys, h, w):
    canvas = np.zeros((h, w), np.uint8)
    if polys:
        cv2.fillPoly(canvas, polys, 1)
    return canvas.astype(bool)


def external_fill(binary):
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros(binary.shape, np.uint8)
    kept = [c for c in contours if keep_contour(c)]
    if kept:
        cv2.drawContours(canvas, kept, -1, 1, cv2.FILLED)
    return canvas.astype(bool)


def holes_fill(binary):
    """Outer contours minus the interior of their holes (2-level RETR_CCOMP)."""
    contours, hierarchy = cv2.findContours(binary.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    shape = binary.shape
    if hierarchy is None:
        return np.zeros(shape, bool)
    hierarchy = hierarchy[0]
    outer = [c for c, hh in zip(contours, hierarchy) if hh[3] == -1 and keep_contour(c)]
    holes = [c for c, hh in zip(contours, hierarchy) if hh[3] != -1]
    filled = np.zeros(shape, np.uint8)
    if outer:
        cv2.drawContours(filled, outer, -1, 1, cv2.FILLED)
    filled = filled.astype(bool)
    if holes:
        hole_area = np.zeros(shape, np.uint8)
        cv2.drawContours(hole_area, holes, -1, 1, cv2.FILLED)
        hole_edge = np.zeros(shape, np.uint8)
        # Suzuki-Abe hole borders run along foreground pixels, so keep the traced edge.
        cv2.drawContours(hole_edge, holes, -1, 1, 1)
        filled &= ~(hole_area.astype(bool) & ~hole_edge.astype(bool))
    return filled


def mask_resolution(mask, imgsz, mask_ratio):
    """Approximate round trip through the prototype-mask grid of a YOLO seg head."""
    h, w = mask.shape
    scale = imgsz / max(h, w)
    ph, pw = max(1, round(h * scale / mask_ratio)), max(1, round(w * scale / mask_ratio))
    low = cv2.resize(mask.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA)
    return cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5


class Score:
    """Pooled confusion counts plus mean Dice over GT-present slices."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0
        self.slice_dice = []

    def add(self, pred, gt):
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
        self.tp += tp; self.fp += fp; self.fn += fn
        if gt.any():
            self.slice_dice.append(2 * tp / (2 * tp + fp + fn))

    def pooled(self):
        denom = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denom if denom else None

    def mean(self):
        return float(np.mean(self.slice_dice)) if self.slice_dice else None


def fmt(v, digits=3):
    return '—' if v is None else f'{v:.{digits}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=REPO / 'configs/experiments/brats2024_sfseg.yaml')
    p.add_argument('--role', default='source_val', choices=('source_val', 'source_train'))
    p.add_argument('--modality', default='t1n', choices=('t1n', 't1c', 't2w', 't2f'))
    p.add_argument('--max-cases', type=int, default=0, help='0 = every case of the role.')
    p.add_argument('--imgsz', type=int, default=None, help='Default: imgsz from --config.')
    p.add_argument('--mask-ratio', type=int, default=4, help='Ultralytics default prototype downsampling.')
    p.add_argument('--smoke', action='store_true', help='Read the smoke prepared folder instead of full.')
    p.add_argument('--out-dir', type=Path, default=None)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    imgsz = args.imgsz or cfg['imgsz']
    suffix = 'smoke' if args.smoke else 'full'
    dst = resolve_path(cfg['prepared_root']) / suffix
    out_dir = args.out_dir or resolve_path(cfg['project']) / suffix / '_label_fidelity' / args.role
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((dst / 'manifest.json').read_text())
    records = [c for c in manifest['cases'] if c['role'] == args.role]
    if args.max_cases:
        records = records[:args.max_cases]
    if not records:
        raise ValueError(f'No {args.role} cases in {dst / "manifest.json"}')
    label_dir = dst / args.modality / 'labels' / args.role

    current = {name: Score() for name in RAW}            # Table A: polygons actually trained on
    current_lowres = {name: Score() for name in RAW}
    lost = defaultdict(int)                              # GT present, no polygon survived
    present = defaultdict(int)
    outside = defaultdict(int)                           # polygon px not of this class in GT
    shared = defaultdict(int)                            # polygon px also covered by another class
    area = defaultdict(int)
    whatif = {(rule, region): Score() for rule in ('external', 'holes', 'holes_lowres') for region in REGIONS}
    missing_labels = slices = 0
    t0 = time.time()

    for ci, case in enumerate(records):
        with np.load(dst / 'evaluation_masks' / args.role / f"{case['case_id']}.npz") as data:
            gt_volume = data['mask']
        h, w = gt_volume.shape[:2]
        for zi, z in enumerate(case['slices']):
            gt = gt_volume[:, :, zi]
            polys = read_polygons(label_dir / f"{case['case_id']}_z{z:03d}.txt", h, w)
            if polys is None:
                missing_labels += 1
                continue
            if not gt.any() and not polys:
                continue                                 # empty slice: contributes nothing
            slices += 1
            rasters = {name: raster(polys.get(raw - 1, []), h, w) for name, raw in RAW.items()}
            for name, raw in RAW.items():
                gt_c, pr_c = gt == raw, rasters[name]
                current[name].add(pr_c, gt_c)
                current_lowres[name].add(mask_resolution(pr_c, imgsz, args.mask_ratio), gt_c)
                if gt_c.any():
                    present[name] += 1
                    lost[name] += int(not pr_c.any())
                others = np.zeros_like(pr_c)
                for other, r in rasters.items():
                    if other != name:
                        others |= r
                area[name] += int(pr_c.sum())
                outside[name] += int((pr_c & ~gt_c).sum())
                shared[name] += int((pr_c & others).sum())
            for region, raws in REGIONS.items():
                gt_r = np.isin(gt, raws)
                if not gt_r.any():
                    continue
                with_holes = holes_fill(gt_r)
                whatif[('external', region)].add(external_fill(gt_r), gt_r)
                whatif[('holes', region)].add(with_holes, gt_r)
                whatif[('holes_lowres', region)].add(mask_resolution(with_holes, imgsz, args.mask_ratio), gt_r)
        if (ci + 1) % 20 == 0 or ci + 1 == len(records):
            print(f'{ci + 1}/{len(records)} cases, {slices} non-empty slices, {time.time() - t0:.0f}s', flush=True)

    if missing_labels:
        print(f'[warn] {missing_labels} slices had no label file in {label_dir}', flush=True)

    rows_a, rows_b = [], []
    for name in RAW:
        rows_a.append(dict(region=name, gt_present_slices=present[name],
                           pooled_dice=current[name].pooled(), mean_slice_dice=current[name].mean(),
                           slices_lost=lost[name],
                           polygon_px_not_this_class=outside[name] / area[name] if area[name] else None,
                           polygon_px_shared_with_other_class=shared[name] / area[name] if area[name] else None,
                           pooled_dice_at_mask_res=current_lowres[name].pooled()))
    for region in REGIONS:
        rows_b.append(dict(region=region,
                           external=whatif[('external', region)].pooled(),
                           holes=whatif[('holes', region)].pooled(),
                           holes_at_mask_res=whatif[('holes_lowres', region)].pooled(),
                           gt_present_slices=len(whatif[('holes', region)].slice_dice)))

    proto = f'{round(imgsz / args.mask_ratio)}px prototype grid (imgsz {imgsz} / mask_ratio {args.mask_ratio}), approximate'
    lines = [f'# BraTS2024 label fidelity — {args.role}, {len(records)} cases, {slices} non-empty slices', '',
             '## A. Polygons the model is trained on vs exact masks', '',
             '| Region | GT-present slices | Pooled Dice | Mean slice Dice | Slices lost | Polygon px not this class | Polygon px shared with other class | Pooled Dice at mask res |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows_a:
        lost_pct = f"{r['slices_lost']} ({100 * r['slices_lost'] / r['gt_present_slices']:.1f}%)" if r['gt_present_slices'] else '0'
        lines.append(f"| {r['region']} | {r['gt_present_slices']} | {fmt(r['pooled_dice'])} | {fmt(r['mean_slice_dice'])} | "
                     f"{lost_pct} | {fmt(r['polygon_px_not_this_class'])} | {fmt(r['polygon_px_shared_with_other_class'])} | "
                     f"{fmt(r['pooled_dice_at_mask_res'])} |")
    lines += ['', f'Mask res = {proto}.', '',
              '## B. Ceiling under alternative label rules (built from GT)', '',
              '| Region | GT-present slices | External contour (current rule) | With holes | With holes at mask res |',
              '|---|---:|---:|---:|---:|']
    for r in rows_b:
        lines.append(f"| {r['region']} | {r['gt_present_slices']} | {fmt(r['external'])} | {fmt(r['holes'])} | {fmt(r['holes_at_mask_res'])} |")

    sanity = [(a['region'], a['pooled_dice'], b['external']) for a, b in zip(rows_a, rows_b)
              if a['pooled_dice'] is not None and b['external'] is not None and abs(a['pooled_dice'] - b['external']) > 0.01]
    lines.append('')
    if sanity:
        lines.append('**Sanity check failed:** Table A and the recomputed external rule disagree for '
                     + ', '.join(f'{n} ({x:.3f} vs {y:.3f})' for n, x, y in sanity)
                     + ' — check polygon orientation/scaling before trusting Table A.')
    else:
        lines.append('Sanity check passed: Table A matches the external rule recomputed from GT.')

    report = '\n'.join(lines) + '\n'
    (out_dir / 'label_fidelity.md').write_text(report)
    (out_dir / 'label_fidelity.json').write_text(json.dumps(dict(
        role=args.role, modality=args.modality, cases=len(records), non_empty_slices=slices,
        imgsz=imgsz, mask_ratio=args.mask_ratio, current_labels=rows_a, alternative_rules=rows_b), indent=2) + '\n')
    print('\n' + report)
    print(f'Saved {out_dir / "label_fidelity.md"} and label_fidelity.json')


if __name__ == '__main__':
    main()
