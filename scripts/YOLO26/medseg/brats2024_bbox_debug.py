#!/usr/bin/env python3
"""Per-slice bbox diagnostic for BraTS2024 MedRT-SFSeg: localize WHERE segmentation fails.

The official evaluate() in brats2024_eval.py only reports voxel Dice/ASD/HD95 per
region. Those numbers (very low on target_test for all four regions, with NETC's
"ok-looking" Dice driven almost entirely by both_empty_cases, see dice_gt_present_mean)
don't say WHY: no box is even proposed near the lesion, a box is proposed but under
the eval confidence (0.25), the box is in the right place but under the wrong class,
or the box is simply mislocated. This script answers that, per class, with numbers
and overlay images, on the sealed target_test split (same split consumed by the
official evaluate()) so it never touches target_train / peeks at adaptation-time labels.

For each requested class it:
  1. Scans GT masks of the first --max-cases-scan target_test (or source_val) cases
     to find the --slices-per-class slices with the LARGEST ground-truth area for
     that class (clean, unambiguous examples -- not noise from tiny specks).
  2. Runs the chosen checkpoint (source_only / adabn / medrt_sfseg) on each selected
     slice at a very low confidence floor (--conf-low) to see every surviving
     candidate box, not just the ones that pass the official eval confidence.
  3. Matches GT boxes (from GT-mask connected components) against ALL predicted boxes
     (any class) by IoU, and records the best match's confidence, predicted class
     and IoU -- regardless of whether that box would pass the eval confidence.
  4. Saves an overlay PNG per slice: GT boxes (focus class in green, other classes in
     gray) vs predicted boxes (orange = passes eval confidence, red = below eval
     confidence but above the viz floor, yellow = the best IoU match even if it is
     below the viz floor, drawn anyway so you can literally see "it found it, but
     wasn't confident").
  5. Writes a CSV of every sampled slice's numbers, and prints a short per-class
     verdict: no-proposal (localization/feature problem) vs low-confidence-but-
     located (calibration problem under domain shift) vs located-but-wrong-class
     (confusion between the four fine-grained sub-regions) vs mislocated.

Usage (run where dataset/BraTS2024-SFSeg + runs/seg/brats2024_sfseg/full checkpoints
both exist -- NOT this Mac, which only has the run outputs, not the prepared dataset):

  python scripts/YOLO26/medseg/brats2024_bbox_debug.py \
      --direction t1n_to_t2w --stage medrt_sfseg --classes NETC ET \
      --slices-per-class 4 --out-dir runs/seg/brats2024_sfseg/full/_bbox_debug/t1n_to_t2w_medrt_sfseg

Compare stages to see if adaptation helped or hurt localization/confidence:

  python scripts/YOLO26/medseg/brats2024_bbox_debug.py --direction t1n_to_t2w --stage source_only --classes NETC ET --out-dir runs/seg/brats2024_sfseg/full/_bbox_debug/t1n_to_t2w_source_only
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from brats2024_data import NAMES  # noqa: E402  (sibling module, same directory)

DIRECTIONS = ('t1n_to_t2w', 't2w_to_t1n', 't1c_to_t2f', 't2f_to_t1c')
STAGE_CHECKPOINT = {
    'source_only': Path('source/weights/best.pt'),
    'adabn': Path('adabn/adabn.pt'),
    'medrt_sfseg': Path('adapt/student_final.pt'),
}
VIZ_COLORS = {  # BGR
    'gt_focus': (0, 255, 0),
    'gt_other': (140, 140, 140),
    'pred_pass_eval': (0, 165, 255),
    'pred_below_eval': (0, 0, 255),
    'pred_best_below_viz': (0, 255, 255),
}


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def boxes_from_mask(mask_slice, class_raw_label):
    """Axis-aligned bbox per connected component of one raw label in a 2D GT slice."""
    binary = (mask_slice == class_raw_label).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0 and h > 0:
            boxes.append((float(x), float(y), float(x + w), float(y + h)))
    return boxes


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def draw_box(img, box, color, label, thickness=1):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        ty = y1 - 4 if y1 - 4 > 10 else y2 + 12
        cv2.putText(img, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=REPO / 'configs/experiments/brats2024_sfseg.yaml')
    p.add_argument('--direction', required=True, choices=DIRECTIONS)
    p.add_argument('--stage', default='medrt_sfseg', choices=tuple(STAGE_CHECKPOINT))
    p.add_argument('--role', default='target_test', choices=('target_test', 'source_val'))
    p.add_argument('--classes', nargs='+', default=list(NAMES.values()), choices=list(NAMES.values()))
    p.add_argument('--slices-per-class', type=int, default=4)
    p.add_argument('--max-cases-scan', type=int, default=40)
    p.add_argument('--conf-low', type=float, default=0.001, help='Floor for counting/matching raw candidates.')
    p.add_argument('--conf-eval', type=float, default=None, help='Default: evaluation.confidence from --config.')
    p.add_argument('--conf-viz-min', type=float, default=0.05, help='Floor for drawing a box (decluttering only).')
    p.add_argument('--iou-match-thresh', type=float, default=0.1, help='IoU above which we call a GT box "localized".')
    p.add_argument('--device', default=None)
    p.add_argument('--imgsz', type=int, default=None)
    p.add_argument('--max-det', type=int, default=300)
    p.add_argument('--weights', help='Explicit checkpoint path, overrides --stage resolution.')
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    conf_eval = args.conf_eval if args.conf_eval is not None else cfg['evaluation']['confidence']
    device = args.device if args.device is not None else cfg.get('device', '0')
    imgsz = args.imgsz if args.imgsz is not None else cfg['imgsz']
    suffix = 'smoke' if cfg.get('smoke') else 'full'

    dst = resolve_path(cfg['prepared_root']) / suffix
    project = resolve_path(cfg['project']) / suffix
    manifest = json.loads((dst / 'manifest.json').read_text())
    run = project / args.direction

    if args.weights:
        checkpoint = resolve_path(args.weights)
    else:
        checkpoint = run / STAGE_CHECKPOINT[args.stage]
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')

    source, target = args.direction.split('_to_')
    modality = source if args.role == 'source_val' else target
    records = [c for c in manifest['cases'] if c['role'] == args.role]
    if not records:
        raise ValueError(f'No cases with role={args.role!r} in manifest')

    name_to_id = {v: k for k, v in NAMES.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: scan GT only (cheap) to pick the cleanest example slices per class ---
    scanned = records[:args.max_cases_scan]
    candidates = {name: [] for name in args.classes}
    gt_cache = {}
    for case in scanned:
        gt_path = dst / 'evaluation_masks' / args.role / f"{case['case_id']}.npz"
        with np.load(gt_path) as data:
            gt = data['mask']
        gt_cache[case['case_id']] = gt
        for name in args.classes:
            raw_label = name_to_id[name] + 1
            areas = (gt == raw_label).sum(axis=(0, 1))
            for zi, area in enumerate(areas):
                if area > 0:
                    candidates[name].append((int(area), case, zi))
    selected = {}
    for name in args.classes:
        candidates[name].sort(key=lambda t: -t[0])
        selected[name] = candidates[name][:args.slices_per_class]
        if len(selected[name]) < args.slices_per_class:
            print(f'[warn] only found {len(selected[name])}/{args.slices_per_class} slices with {name} '
                  f'present in the first {args.max_cases_scan} {args.role} cases; raise --max-cases-scan.')

    # --- Pass 2: inference on just the selected slices ---
    from ultralytics import YOLO
    model = YOLO(str(checkpoint), task='segment')

    rows = []
    for name in args.classes:
        raw_label = name_to_id[name] + 1
        for area, case, zi in selected[name]:
            case_id, z = case['case_id'], case['slices'][zi]
            gt = gt_cache[case_id]
            gt_slice = gt[:, :, zi]
            gt_boxes_focus = boxes_from_mask(gt_slice, raw_label)
            image_path = dst / modality / 'images' / args.role / f'{case_id}_z{z:03d}.png'

            result = model.predict(source=str(image_path), imgsz=imgsz, device=device, conf=args.conf_low,
                                    max_det=args.max_det, retina_masks=False, verbose=False)[0]
            pred_boxes = result.boxes.xyxy.detach().cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
            pred_cls = result.boxes.cls.detach().cpu().numpy().astype(int) if result.boxes is not None else np.zeros((0,), dtype=int)
            pred_conf = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.zeros((0,))

            best_iou, best_idx = 0.0, None
            for gt_box in gt_boxes_focus:
                for pi, pbox in enumerate(pred_boxes):
                    iou = iou_xyxy(gt_box, tuple(pbox))
                    if iou > best_iou:
                        best_iou, best_idx = iou, pi

            # Same-class match: the any-class best-IoU box can be a junk low-conf box of
            # another class while a correct-class box sits at slightly lower IoU.
            focus_id = name_to_id[name]
            same_iou, same_idx = 0.0, None
            for gt_box in gt_boxes_focus:
                for pi, pbox in enumerate(pred_boxes):
                    if int(pred_cls[pi]) != focus_id:
                        continue
                    iou = iou_xyxy(gt_box, tuple(pbox))
                    if iou > same_iou:
                        same_iou, same_idx = iou, pi
            focus_confs = pred_conf[pred_cls == focus_id]
            max_focus_conf = float(focus_confs.max()) if len(focus_confs) else None

            best_conf = float(pred_conf[best_idx]) if best_idx is not None else None
            best_class = NAMES[int(pred_cls[best_idx])] if best_idx is not None else None
            n_pass_eval = int((pred_conf >= conf_eval).sum())
            row = dict(direction=args.direction, stage=args.stage, role=args.role, case_id=case_id, z=int(z),
                       focus_class=name, gt_area_px=int(area), n_gt_boxes=len(gt_boxes_focus),
                       n_pred_low_conf=int(len(pred_boxes)), n_pred_pass_eval=n_pass_eval,
                       best_iou=round(best_iou, 4), best_match_conf=best_conf, best_match_class=best_class,
                       class_correct=(best_class == name) if best_class else None,
                       localized=best_iou >= args.iou_match_thresh,
                       same_class_iou=round(same_iou, 4),
                       same_class_conf=float(pred_conf[same_idx]) if same_idx is not None else None,
                       max_focus_class_conf_anywhere=max_focus_conf)
            rows.append(row)

            # --- overlay image ---
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            for other_label, other_name in NAMES.items():
                if other_name == name:
                    continue
                for box in boxes_from_mask(gt_slice, other_label + 1):
                    draw_box(img, box, VIZ_COLORS['gt_other'], other_name, thickness=1)
            for box in gt_boxes_focus:
                draw_box(img, box, VIZ_COLORS['gt_focus'], f'GT:{name}', thickness=2)
            for pi, pbox in enumerate(pred_boxes):
                conf = float(pred_conf[pi])
                if conf < args.conf_viz_min and pi != best_idx:
                    continue
                pred_name = NAMES[int(pred_cls[pi])]
                color = VIZ_COLORS['pred_pass_eval'] if conf >= conf_eval else VIZ_COLORS['pred_below_eval']
                if pi == best_idx and conf < args.conf_viz_min:
                    color = VIZ_COLORS['pred_best_below_viz']
                draw_box(img, pbox, color, f'{pred_name} {conf:.2f}', thickness=2 if conf >= conf_eval else 1)
            out_path = args.out_dir / f'{args.direction}_{args.stage}_{name}_{case_id}_z{z:03d}.png'
            cv2.imwrite(str(out_path), img)

    # --- CSV ---
    csv_path = args.out_dir / 'bbox_debug_summary.csv'
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {len(rows)} rows -> {csv_path}')
    print(f'Overlay PNGs -> {args.out_dir}')

    # --- per-class verdict ---
    print(f'\n=== Verdict (direction={args.direction}, stage={args.stage}, role={args.role}, '
          f'conf_eval={conf_eval}, iou_match_thresh={args.iou_match_thresh}) ===')
    for name in args.classes:
        sub = [r for r in rows if r['focus_class'] == name]
        if not sub:
            continue
        n = len(sub)
        n_localized = sum(r['localized'] for r in sub)
        n_no_proposal = sum(r['n_pred_low_conf'] == 0 for r in sub)
        n_localized_conf_pass = sum(r['localized'] and r['best_match_conf'] is not None and r['best_match_conf'] >= conf_eval for r in sub)
        n_localized_class_correct = sum(r['localized'] and r['class_correct'] for r in sub)
        print(f'\n[{name}] {n} slices (GT area {min(r["gt_area_px"] for r in sub)}-{max(r["gt_area_px"] for r in sub)} px)')
        print(f'  localized (best IoU >= {args.iou_match_thresh}): {n_localized}/{n}')
        print(f'  zero candidate boxes at conf>={args.conf_low}: {n_no_proposal}/{n}')
        if n_localized:
            print(f'  of localized: confidence >= eval threshold ({conf_eval}): {n_localized_conf_pass}/{n_localized}')
            print(f'  of localized: predicted class correct: {n_localized_class_correct}/{n_localized}')
            confs = [r['best_match_conf'] for r in sub if r['localized'] and r['best_match_conf'] is not None]
            if confs:
                print(f'  of localized: best-match confidence mean={np.mean(confs):.3f} median={np.median(confs):.3f}')
        same_ok = [r for r in sub if r['same_class_iou'] >= 0.5]
        same_ok_pass = [r for r in same_ok if r['same_class_conf'] >= conf_eval]
        print(f'  same-class box with IoU >= 0.5: {len(same_ok)}/{n}; of those conf >= {conf_eval}: {len(same_ok_pass)}/{len(same_ok) or 1}')
        sc = [r['same_class_conf'] for r in sub if r['same_class_conf'] is not None]
        mx = [r['max_focus_class_conf_anywhere'] for r in sub if r['max_focus_class_conf_anywhere'] is not None]
        if sc:
            print(f'  same-class best-IoU box conf: median={np.median(sc):.3f}; max {name} conf anywhere in slice: median={np.median(mx):.3f}')
        context = ('in-domain (no modality shift)' if args.role == 'source_val'
                   else f'target domain, stage={args.stage}')
        if len(same_ok) == 0:
            print(f'  -> DIAGNOSIS ({context}): no confident-or-not {name} box tightly covers the lesion '
                  '(same-class IoU >= 0.5 never reached). The model has not learned to localize this '
                  'class here; thresholds/calibration cannot fix it.')
        elif len(same_ok_pass) < len(same_ok):
            print(f'  -> DIAGNOSIS ({context}): correct-class boxes exist on the lesion but score below '
                  f'{conf_eval}. Low confidence for this class'
                  + (' even without domain shift -> source training problem.' if args.role == 'source_val'
                     else '; compare with --role source_val to see if it is shift-induced.'))
        else:
            print(f'  -> DIAGNOSIS ({context}): correct-class, confident boxes on these examples; '
                  'look at other slices/cases for the failure.')


if __name__ == '__main__':
    main()
