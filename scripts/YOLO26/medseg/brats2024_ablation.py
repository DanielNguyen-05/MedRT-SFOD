#!/usr/bin/env python3
"""BraTS2024 ablations on top of an existing run: drop AdaBN and/or invert target intensity.

Kept outside brats2024_experiments.py on purpose. That pipeline hashes its own source code
into every stage's protocol, so editing it would invalidate the finished runs. This runner
reuses the finished source checkpoint (`<run>/source/weights/best.pt`) and the run's own
`effective_config.yaml`, then writes everything under `<run>/ablations/<variant>/`, so the
main run is never touched.

Variants
  no_adabn         Mean Teacher + Mask-DHF/DURR + SegMARD start from the source checkpoint.
                   Note: only the explicit AdaBN step is removed. The Student still trains in
                   train() mode, so its BN running statistics drift toward the target during
                   adaptation; the Teacher keeps the source BN buffers (EMA covers parameters
                   only, see update_teacher_ema).
  invert           Target images are intensity-inverted inside the brain (x -> 256 - x for
                   x > 0, background stays 0), then the normal AdaBN -> adapt chain runs.
                   The source-only model is also evaluated on inverted target_test: that
                   number alone says whether inversion moves the target closer to the source.
  invert_no_adabn  Both changes.

Why inversion: T1-weighted and T2-weighted images order the tissues roughly in reverse
(CSF dark on T1, bright on T2; white matter brighter than grey on T1, darker on T2), so for
t1n<->t2w a brain-masked inversion is a label-free, source-free way to bring target intensity
closer to the source. FLAIR suppresses CSF, so for t1c<->t2f the inversion is only partial;
treat that as an empirical ablation. Only target images are transformed; no target label is
read before the final target_test evaluation, and no source data is used.

  python scripts/YOLO26/medseg/brats2024_ablation.py --direction t1n_to_t2w --variant invert --steps source_eval
  nohup python -u scripts/YOLO26/medseg/brats2024_ablation.py --direction t1n_to_t2w --variant no_adabn \
      > logs/brats2024/ablation_t1n_to_t2w_no_adabn.log 2>&1 &
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from brats2024_data import NAMES, write_json  # noqa: E402

DIRECTIONS = ('t1n_to_t2w', 't2w_to_t1n', 't1c_to_t2f', 't2f_to_t1c')
VARIANTS = {'no_adabn': dict(invert=False, adabn=False),
            'invert': dict(invert=True, adabn=True),
            'invert_no_adabn': dict(invert=True, adabn=False)}
STEPS = ('prepare_images', 'source_eval', 'adabn', 'adapt', 'evaluate', 'summarize')
INVERT_RULE = 'brain-masked inversion: x -> 256 - x for x > 0, 0 stays 0 (uint8)'


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def invert_image(gray):
    """1..255 -> 255..1 inside the brain; background (0) stays 0, so the brain mask is preserved."""
    out = np.zeros_like(gray)
    brain = gray > 0
    out[brain] = (256 - gray[brain].astype(np.int16)).astype(np.uint8)
    return out


def derive_inverted(dst, manifest, target, roles=('target_train', 'target_test')):
    """Write <dst>/<target>_inv/images/<role>/*.png from the prepared target images (images only)."""
    out_root = dst / f'{target}_inv'
    marker = out_root / 'derived.json'
    expected = {role: sum(len(c['slices']) for c in manifest['cases'] if c['role'] == role) for role in roles}
    if marker.is_file():
        info = json.loads(marker.read_text())
        if info.get('rule') == INVERT_RULE and info.get('counts') == expected \
                and info.get('split_sha256') == manifest['split_sha256']:
            print(f'Inverted images already derived: {out_root}', flush=True)
            return out_root
    t0 = time.time()
    for role in roles:
        src_dir, out_dir = dst / target / 'images' / role, out_root / 'images' / role
        out_dir.mkdir(parents=True, exist_ok=True)
        done = 0
        cases = [c for c in manifest['cases'] if c['role'] == role]
        for ci, case in enumerate(cases):
            for z in case['slices']:
                name = f"{case['case_id']}_z{z:03d}.png"
                gray = cv2.imread(str(src_dir / name), cv2.IMREAD_UNCHANGED)
                if gray is None or gray.ndim != 2 or gray.dtype != np.uint8:
                    raise ValueError(f'Expected a single-channel uint8 PNG: {src_dir / name}')
                if not cv2.imwrite(str(out_dir / name), invert_image(gray)):
                    raise IOError(f'PNG write failed: {out_dir / name}')
                done += 1
            if (ci + 1) % 50 == 0 or ci + 1 == len(cases):
                print(f'invert {role}: {done}/{expected[role]} slices, {time.time() - t0:.0f}s', flush=True)
        actual = len(list(out_dir.glob('*.png')))
        if actual != expected[role]:
            raise ValueError(f'{out_dir}: {actual} PNGs, expected {expected[role]}')
    write_json(marker, dict(rule=INVERT_RULE, source_modality=target, counts=expected,
                            split_sha256=manifest['split_sha256'], target_labels_used=False))
    return out_root


def load_model(checkpoint):
    from ultralytics import YOLO
    model = YOLO(str(checkpoint), task='segment')
    if model.model.model[-1].nc != 4 or not model.model.end2end:
        raise ValueError('Evaluation requires a native end-to-end four-class Segment26')
    return model


def run_eval(checkpoint, dst, manifest, modality, cfg, output, dry):
    if output.is_file():
        print(f'Skip existing evaluation: {output}', flush=True)
        return
    print(f'Evaluate {checkpoint} on {modality}/target_test -> {output}', flush=True)
    if dry:
        return
    from brats2024_eval import evaluate
    evaluate(load_model(checkpoint), dst, manifest, modality, 'target_test', cfg, output, checkpoint)


def run_train(mode, weights, images, out_dir, config_file, resume, dry):
    command = [sys.executable, str(Path(__file__).with_name('brats2024_train.py')), '--mode', mode,
               '--weights', str(weights), '--target-images', str(images), '--out-dir', str(out_dir),
               '--config', str(config_file)]
    if resume:
        command.append('--resume')
    print('$ ' + ' '.join(command), flush=True)
    if not dry:
        subprocess.run(command, cwd=REPO, check=True)


def fmt(v):
    return '—' if v is None else f'{v:.3f}'


def summarize(run, abl, variant):
    rows = [('source_only', run / 'evaluation' / 'source_only_target_test.json'),
            ('adabn', run / 'evaluation' / 'adabn_target_test.json'),
            ('medrt_sfseg', run / 'evaluation' / 'medrt_sfseg_target_test.json')]
    rows = [(f'main / {s}', p) for s, p in rows]
    rows += [(f'{variant} / {s}', abl / 'evaluation' / f'{s}_target_test.json')
             for s in ('source_only', 'adabn', 'medrt_sfseg')]
    rows = [(label, json.loads(p.read_text())['summary']) for label, p in rows if p.is_file()]
    names = list(NAMES.values())
    lines = [f'# {run.name} — ablation `{variant}` vs main run (target_test, conf from effective_config)', '']
    for key, title in (('dice_gt_present_mean', 'Dice on GT-present cases'), ('dice', 'Dice, official empty policy'),
                       ('hd95_mm', 'HD95 (mm, defined cases)')):
        lines += [f'## {title}', '', '| Run / stage | ' + ' | '.join(names) + ' | mean |',
                  '|---|' + '---:|' * (len(names) + 1)]
        for label, summary in rows:
            vals = [summary[n][key] if key == 'dice_gt_present_mean' else summary[n][key]['case_mean'] for n in names]
            valid = [v for v in vals if v is not None]
            lines.append(f'| {label} | ' + ' | '.join(fmt(v) for v in vals) + f' | {fmt(np.mean(valid) if valid else None)} |')
        lines.append('')
    (abl / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path, default=REPO / 'configs/experiments/brats2024_sfseg.yaml',
                   help='Only used to locate the project; hyperparameters come from <run>/effective_config.yaml.')
    p.add_argument('--direction', required=True, choices=DIRECTIONS)
    p.add_argument('--variant', required=True, choices=tuple(VARIANTS))
    p.add_argument('--steps', nargs='+', choices=STEPS, default=list(STEPS))
    p.add_argument('--device', default=None)
    p.add_argument('--adapt-epochs', type=int, default=None, help='Pilot only; recorded in ablation.json.')
    p.add_argument('--dry-run', action='store_true', help='Print what would run, train/evaluate nothing.')
    args = p.parse_args()

    base = yaml.safe_load(args.config.read_text())
    variant = VARIANTS[args.variant]
    run = resolve_path(base['project']) / 'full' / args.direction
    run_cfg = run / 'effective_config.yaml'
    if not run_cfg.is_file():
        raise FileNotFoundError(f'{run_cfg} (run the main adaptation for this direction first)')
    cfg = yaml.safe_load(run_cfg.read_text())
    if cfg.get('smoke'):
        raise ValueError('Ablations are defined on full runs only')
    if args.device is not None:
        cfg['device'] = args.device
    if args.adapt_epochs is not None:
        cfg['adaptation']['epochs'] = args.adapt_epochs
    dst = resolve_path(cfg['prepared_root']) / 'full'
    manifest = json.loads((dst / 'manifest.json').read_text())
    source_ckpt = run / 'source' / 'weights' / 'best.pt'
    if not (run / 'source' / 'complete.json').is_file() or not source_ckpt.is_file():
        raise FileNotFoundError('Finished source stage required: ' + str(source_ckpt))
    _, target = args.direction.split('_to_')
    modality = f'{target}_inv' if variant['invert'] else target

    abl = run / 'ablations' / args.variant
    abl.mkdir(parents=True, exist_ok=True)
    config_file = abl / 'effective_config.yaml'
    config_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    record = dict(variant=args.variant, **variant, target_modality_dir=modality,
                  intensity_rule=INVERT_RULE if variant['invert'] else None,
                  source_checkpoint=str(source_ckpt), source_sha256=file_hash(source_ckpt),
                  adapt_epochs=cfg['adaptation']['epochs'], evaluation_confidence=cfg['evaluation']['confidence'],
                  split_sha256=manifest['split_sha256'], target_labels_used_before_final_eval=False)
    previous = abl / 'ablation.json'
    if previous.is_file():
        old = json.loads(previous.read_text())
        changed = [k for k in ('source_sha256', 'adapt_epochs', 'target_modality_dir', 'split_sha256') if old.get(k) != record[k]]
        if changed:
            raise ValueError(f'{abl} was produced with different {changed}; move it away before rerunning')
    write_json(previous, record)
    images = dst / modality / 'images' / 'target_train'

    if 'prepare_images' in args.steps and variant['invert'] and not args.dry_run:
        derive_inverted(dst, manifest, target)
    if variant['invert'] and not args.dry_run and not images.is_dir():
        raise FileNotFoundError(f'{images}; include the prepare_images step')

    if 'source_eval' in args.steps and variant['invert']:
        run_eval(source_ckpt, dst, manifest, modality, cfg, abl / 'evaluation' / 'source_only_target_test.json', args.dry_run)

    adapt_weights = source_ckpt
    if variant['adabn']:
        adabn_ckpt = abl / 'adabn' / 'adabn.pt'
        if 'adabn' in args.steps and not adabn_ckpt.is_file():
            run_train('adabn', source_ckpt, images, abl / 'adabn', config_file, False, args.dry_run)
        adapt_weights = adabn_ckpt

    student = abl / 'adapt' / 'student_final.pt'
    if 'adapt' in args.steps and not student.is_file():
        if not args.dry_run and not adapt_weights.is_file():
            raise FileNotFoundError(f'{adapt_weights}; run the adabn step first')
        run_train('adapt', adapt_weights, images, abl / 'adapt', config_file,
                  (abl / 'adapt' / 'resume.pt').is_file(), args.dry_run)

    if 'evaluate' in args.steps:
        if variant['adabn'] and (args.dry_run or (abl / 'adabn' / 'adabn.pt').is_file()):
            run_eval(abl / 'adabn' / 'adabn.pt', dst, manifest, modality, cfg,
                     abl / 'evaluation' / 'adabn_target_test.json', args.dry_run)
        if args.dry_run or student.is_file():
            run_eval(student, dst, manifest, modality, cfg, abl / 'evaluation' / 'medrt_sfseg_target_test.json', args.dry_run)

    if 'summarize' in args.steps and not args.dry_run:
        summarize(run, abl, args.variant)


if __name__ == '__main__':
    main()
