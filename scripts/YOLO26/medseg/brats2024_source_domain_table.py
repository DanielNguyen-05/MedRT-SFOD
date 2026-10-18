#!/usr/bin/env python3
"""Source-domain (in-domain) tables for BraTS2024, next to the cross-domain results.

Table 1  In-domain reference: each direction's source model on its own source_val modality
         (already produced by the main pipeline as evaluation/source_val.json).
Table 2  Domain gap: the same source model on source_val vs on target_test, Dice on GT-present
         cases, so the drop caused by the modality shift is explicit.
Table 3  (--evaluate-adapted) Source-domain retention: AdaBN and the final MedRT-SFSeg
         Student evaluated back on source_val. Shows whether adaptation forgets the source.
         Results go to <run>/evaluation_source_domain/, never into the protocol-tracked
         <run>/evaluation/ folder.

Writes Markdown and a LaTeX (booktabs) version under <project>/full/_source_domain/.

  python scripts/YOLO26/medseg/brats2024_source_domain_table.py
  python scripts/YOLO26/medseg/brats2024_source_domain_table.py --evaluate-adapted
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from brats2024_data import NAMES  # noqa: E402

DIRECTIONS = ('t1n_to_t2w', 't2w_to_t1n', 't1c_to_t2f', 't2f_to_t1c')
MOD = {'t1n': 'T1n', 't1c': 'T1c', 't2w': 'T2w', 't2f': 'T2f'}


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def load(path):
    return json.loads(path.read_text())['summary'] if path.is_file() else None


def cell(summary, region, key):
    if summary is None:
        return None
    s = summary[region]
    return s['dice_gt_present_mean'] if key == 'dice_gt' else s[key]['case_mean']


def f(v, scale=100.0):
    return '—' if v is None else f'{v * scale:.1f}'


def mean(values):
    v = [x for x in values if x is not None]
    return float(np.mean(v)) if v else None


def evaluate_on_source(run, stage, checkpoint, dst, manifest, source, cfg):
    out = run / 'evaluation_source_domain' / f'{stage}_source_val.json'
    if out.is_file() or not checkpoint.is_file():
        return out
    from ultralytics import YOLO
    from brats2024_eval import evaluate
    model = YOLO(str(checkpoint), task='segment')
    if model.model.model[-1].nc != 4 or not model.model.end2end:
        raise ValueError('Evaluation requires a native end-to-end four-class Segment26')
    print(f'Evaluate {stage} on {source}/source_val -> {out}', flush=True)
    evaluate(model, dst, manifest, source, 'source_val', cfg, out, checkpoint)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=REPO / 'configs/experiments/brats2024_sfseg.yaml')
    p.add_argument('--directions', nargs='+', choices=DIRECTIONS, default=list(DIRECTIONS))
    p.add_argument('--evaluate-adapted', action='store_true',
                   help='Also evaluate adabn.pt and student_final.pt on source_val (GPU, ~minutes per model).')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    base = yaml.safe_load(args.config.read_text())
    project = resolve_path(base['project']) / 'full'
    dst = resolve_path(base['prepared_root']) / 'full'
    names = list(NAMES.values())
    out_dir = project / '_source_domain'
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((dst / 'manifest.json').read_text()) if args.evaluate_adapted else None

    rows = []
    for d in args.directions:
        run = project / d
        source, target = d.split('_to_')
        src_val = load(run / 'evaluation' / 'source_val.json')
        tgt = load(run / 'evaluation' / 'source_only_target_test.json')
        adapted = {}
        if args.evaluate_adapted:
            cfg_file = run / 'effective_config.yaml'
            cfg = yaml.safe_load(cfg_file.read_text()) if cfg_file.is_file() else dict(base)
            if args.device is not None:
                cfg['device'] = args.device
            for stage, ckpt in (('adabn', run / 'adabn' / 'adabn.pt'), ('medrt_sfseg', run / 'adapt' / 'student_final.pt')):
                adapted[stage] = load(evaluate_on_source(run, stage, ckpt, dst, manifest, source, cfg))
        else:
            for stage in ('adabn', 'medrt_sfseg'):
                adapted[stage] = load(run / 'evaluation_source_domain' / f'{stage}_source_val.json')
        rows.append(dict(direction=d, source=source, target=target, src_val=src_val, tgt=tgt, **adapted))

    md, tex = [], []
    # Table 1 -------------------------------------------------------------------------------
    md += ['# BraTS2024 source-domain tables', '',
           '## Table 1. In-domain reference: source model on its own modality (source_val)', '',
           'Dice (%) on GT-present cases / HD95 (mm). Mean over the four regions.', '',
           '| Direction | Source modality | ' + ' | '.join(f'{n} Dice' for n in names) + ' | Mean Dice | '
           + ' | '.join(f'{n} HD95' for n in names) + ' |',
           '|---|---|' + '---:|' * (2 * len(names) + 1)]
    for r in rows:
        dice = [cell(r['src_val'], n, 'dice_gt') for n in names]
        hd = [cell(r['src_val'], n, 'hd95_mm') for n in names]
        md.append(f"| {r['direction']} | {MOD[r['source']]} | " + ' | '.join(f(v) for v in dice) + f" | {f(mean(dice))} | "
                  + ' | '.join(f(v, 1) for v in hd) + ' |')
    # Table 2 -------------------------------------------------------------------------------
    md += ['', '## Table 2. Domain gap of the source model (Dice % on GT-present cases)', '',
           '| Direction | Evaluated on | ' + ' | '.join(names) + ' | Mean |', '|---|---|' + '---:|' * (len(names) + 1)]
    for r in rows:
        a = [cell(r['src_val'], n, 'dice_gt') for n in names]
        b = [cell(r['tgt'], n, 'dice_gt') for n in names]
        md.append(f"| {r['direction']} | {MOD[r['source']]} (source_val) | " + ' | '.join(f(v) for v in a) + f' | {f(mean(a))} |')
        md.append(f"| | {MOD[r['target']]} (target_test) | " + ' | '.join(f(v) for v in b) + f' | {f(mean(b))} |')
        gap = [None if x is None or y is None else y - x for x, y in zip(a, b)]
        md.append('| | Δ | ' + ' | '.join('—' if g is None else f'{100 * g:+.1f}' for g in gap)
                  + f" | {'—' if mean(gap) is None else f'{100 * mean(gap):+.1f}'} |")
    # Table 3 -------------------------------------------------------------------------------
    if any(r['adabn'] or r['medrt_sfseg'] for r in rows):
        md += ['', '## Table 3. Source-domain retention after adaptation (source_val, Dice % on GT-present cases)', '',
               '| Direction | Model | ' + ' | '.join(names) + ' | Mean |', '|---|---|' + '---:|' * (len(names) + 1)]
        for r in rows:
            for label, key in (('Source only', 'src_val'), ('AdaBN', 'adabn'), ('MedRT-SFSeg', 'medrt_sfseg')):
                v = [cell(r[key], n, 'dice_gt') for n in names]
                md.append(f"| {r['direction'] if label == 'Source only' else ''} | {label} | "
                          + ' | '.join(f(x) for x in v) + f' | {f(mean(v))} |')
    md += ['', 'Notes: patient-disjoint split; Dice on GT-present cases excludes empty-vs-empty matches; '
           'HD95 averages only cases where both masks are non-empty. Not comparable with slice-wise BraTS protocols.', '']

    # LaTeX (Tables 1 and 2) ---------------------------------------------------------------
    tex += [r'% requires \usepackage{booktabs,multirow}',
            r'\begin{table}[t]\centering\small',
            r'\caption{Source-domain reference and domain gap of the source model on BraTS2024 '
            r'(Dice \%, GT-present cases; patient-disjoint split).}',
            r'\label{tab:brats_source_domain}',
            r'\begin{tabular}{ll' + 'c' * (len(names) + 1) + '}', r'\toprule',
            r'Direction & Evaluated on & ' + ' & '.join(names) + r' & Mean \\', r'\midrule']
    for i, r in enumerate(rows):
        a = [cell(r['src_val'], n, 'dice_gt') for n in names]
        b = [cell(r['tgt'], n, 'dice_gt') for n in names]
        d = r['direction'].replace('_to_', r'$\rightarrow$').replace('t1n', 'T1n').replace('t1c', 'T1c') \
            .replace('t2w', 'T2w').replace('t2f', 'T2f')
        tex.append(rf"\multirow{{2}}{{*}}{{{d}}} & {MOD[r['source']]} (source) & " + ' & '.join(f(v) for v in a)
                   + rf' & {f(mean(a))} \\')
        tex.append(rf" & {MOD[r['target']]} (target) & " + ' & '.join(f(v) for v in b) + rf' & {f(mean(b))} \\')
        if i + 1 < len(rows):
            tex.append(r'\midrule')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']

    (out_dir / 'source_domain_tables.md').write_text('\n'.join(md))
    (out_dir / 'source_domain_tables.tex').write_text('\n'.join(tex).replace('—', '--'))
    print('\n'.join(md))
    print(f'Saved {out_dir / "source_domain_tables.md"} and source_domain_tables.tex')


if __name__ == '__main__':
    main()
