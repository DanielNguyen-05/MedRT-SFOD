#!/usr/bin/env python3
"""Run independent MedRT-SFSeg modality-shift experiments on BraTS2024 GLI."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import yaml

from brats2024_data import make_plan, prepare, verify_prepared, source_yaml, digest, write_json

DIRECTIONS = ('t1n_to_t2w', 't2w_to_t1n', 't1c_to_t2f', 't2f_to_t1c')


def resolve_path(path):
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def code_signature():
    paths = [Path(__file__), Path(__file__).with_name('brats2024_data.py'),
             Path(__file__).with_name('brats2024_train.py'), Path(__file__).with_name('brats2024_eval.py'),
             Path(__file__).with_name('durr_multiclass.py'), Path(__file__).with_name('durr_seg.py'),
             Path(__file__).with_name('segmard_seg.py'), Path(__file__).with_name('smoke_stage2_mt_seg.py'),
             Path(__file__).with_name('stage1_adabn_seg.py'),
             Path(__file__).with_name('stage2_dense_sfseg_durr.py'), REPO / 'ultralytics/utils/loss.py']
    return digest({str(p.relative_to(REPO)): file_hash(p) for p in paths})


def effective_config(args):
    cfg = yaml.safe_load(args.config.read_text())
    for key in ('device', 'batch', 'workers', 'imgsz', 'seed', 'raw_root', 'prepared_root', 'project'):
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    if args.weights:
        cfg['source']['weights'] = args.weights
    if args.source_epochs is not None:
        cfg['source']['epochs'] = args.source_epochs
    if args.adapt_epochs is not None:
        cfg['adaptation']['epochs'] = args.adapt_epochs
    cfg['smoke'] = args.smoke
    if args.smoke:
        cfg.update(imgsz=128, batch=2, workers=0, adabn_epochs=1)
        if args.device is None:
            cfg['device'] = 'cpu'
        cfg['source'].update(weights='yolo26n-seg.yaml', epochs=1, patience=0)
        cfg['adaptation'].update(epochs=1, calibration_images=0, hall_student_threshold=.1,
                                 hall_area_threshold=.01, warmup_epochs=.1)
        # A randomly initialized smoke source cannot supply useful pseudo labels.
        # Low hall gating exercises real label-free backward; never use these
        # smoke hyperparameters or scores for scientific conclusions.
    if min(cfg['imgsz'], cfg['batch'], cfg['source']['epochs'], cfg['adaptation']['epochs'], cfg['adabn_epochs']) < 1:
        raise ValueError('Image size, batch, and epoch counts must be positive')
    if cfg['imgsz'] % 32 or cfg['workers'] < 0:
        raise ValueError('imgsz must be divisible by 32 and workers nonnegative')
    if not 0 < cfg['evaluation']['confidence'] < 1:
        raise ValueError('Evaluation confidence must be between 0 and 1')
    return cfg


def check_stage(directory, protocol, resume=False):
    """Idempotent completion; refuse silent restarts or changed experiment settings."""
    path = directory / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError(f'Changed experiment protocol; choose a new --project: {directory}')
        done = directory / 'complete.json'
        if done.exists():
            for name, expected in json.loads(done.read_text())['artifacts'].items():
                if not (directory / name).is_file() or file_hash(directory / name) != expected:
                    raise ValueError(f'Completed stage artifact changed/missing: {directory / name}')
            print(f'Skip completed stage: {directory}', flush=True)
            return False
        if not resume:
            raise FileExistsError(f'Incomplete stage: {directory}. Use --resume to continue/retry.')
    else:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(directory)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(path, protocol)
    return True


def mark_complete(directory, artifacts):
    write_json(directory / 'complete.json', dict(artifacts={name: file_hash(directory / name) for name in artifacts}))


def train_source(dst, manifest, source, directory, cfg, resume):
    from ultralytics import YOLO
    protocol = dict(split=manifest['split_sha256'], source_modality=source,
                    implementation_sha256=code_signature(),
                    config={k: cfg[k] for k in ('source', 'seed', 'imgsz', 'batch', 'smoke')})
    weights = resolve_path(cfg['source']['weights'])
    if not weights.is_file() and str(weights).endswith('.pt'):
        # Resolve/download the initialization before recording its identity.
        # Otherwise the first run records "missing" and the next sees a hash.
        initial = YOLO(cfg['source']['weights'], task='segment')
        weights = Path(initial.ckpt_path).resolve()
        del initial
    protocol['initial_weights_hash'] = file_hash(weights) if weights.is_file() else None
    if not check_stage(directory, protocol, resume):
        return
    data = source_yaml(dst, source, directory / 'source_dataset.yaml')
    last = directory / 'weights' / 'last.pt'
    if resume and last.exists():
        model = YOLO(str(last), task='segment')
        # Native completed checkpoints are stripped; recover an interrupted
        # finalization without attempting to resume a finished optimizer.
        import torch
        state = torch.load(last, map_location='cpu', weights_only=False)
        already_finished = state.get('epoch', -1) == -1 or state.get('epoch', -1) + 1 >= cfg['source']['epochs']
        if not already_finished:
            model.train(resume=True, device=cfg['device'], workers=cfg['workers'])
    else:
        model = YOLO(str(weights) if weights.is_file() else cfg['source']['weights'], task='segment')
        model.train(data=data, epochs=cfg['source']['epochs'], patience=cfg['source']['patience'],
                    imgsz=cfg['imgsz'], batch=cfg['batch'], workers=cfg['workers'], device=cfg['device'],
                    seed=cfg['seed'], deterministic=True, project=str(directory.parent), name=directory.name,
                    exist_ok=True, optimizer='AdamW', lr0=cfg['source']['lr'],
                    hsv_h=0., hsv_s=0., hsv_v=0., bgr=0., mosaic=0., mixup=0., copy_paste=0., close_mosaic=0,
                    fliplr=.5, flipud=0., degrees=10., translate=.05, scale=.1,
                    overlap_mask=False, amp=False, plots=False, save=True)
    best = directory / 'weights' / 'best.pt'
    if not best.is_file():
        raise FileNotFoundError(best)
    write_json(directory / 'selection.json', dict(checkpoint=str(best),
                criterion='maximum source-val box+mask mAP50-95 fitness (native Ultralytics)',
                target_images_used=False, target_labels_used=False))
    mark_complete(directory, ['weights/best.pt', 'selection.json'])


def run_adaptation(dst, manifest, target, run, cfg, resume):
    source_ckpt = run / 'source' / 'weights' / 'best.pt'
    if not (run / 'source' / 'complete.json').is_file():
        raise FileNotFoundError('Complete the source stage before adaptation')
    config_file = run / 'effective_config.yaml'
    config_file.write_text(yaml.safe_dump(cfg, sort_keys=False))
    target_images = dst / target / 'images' / 'target_train'
    for stage, weights, artifacts in (
        ('adabn', source_ckpt, ['adabn.pt', 'adabn.json']),
        ('adapt', run / 'adabn' / 'adabn.pt', ['student_final.pt', 'teacher_final.pt', 'adaptation.json']),
    ):
        directory = run / stage
        protocol = dict(split=manifest['split_sha256'], target_modality=target, input_sha256=file_hash(weights),
                        implementation_sha256=code_signature(),
                        config={k: cfg[k] for k in ('seed', 'imgsz', 'batch', 'adabn_epochs', 'adaptation', 'smoke')})
        if not check_stage(directory, protocol, resume):
            continue
        command = [sys.executable, str(Path(__file__).with_name('brats2024_train.py')),
                   '--mode', stage, '--weights', str(weights), '--target-images', str(target_images),
                   '--out-dir', str(directory), '--config', str(config_file)]
        if stage == 'adapt' and resume and (directory / 'resume.pt').exists():
            command.append('--resume')
        subprocess.run(command, cwd=REPO, check=True)
        mark_complete(directory, artifacts)


def run_evaluation(dst, manifest, source, target, run, cfg, resume):
    from ultralytics import YOLO
    from brats2024_eval import evaluate
    if not (run / 'adapt' / 'complete.json').is_file():
        raise ValueError('Finish adaptation and freeze final Student before opening target test labels')
    checkpoints = dict(source_only=run / 'source/weights/best.pt', adabn=run / 'adabn/adabn.pt',
                       medrt_sfseg=run / 'adapt/student_final.pt')
    directory = run / 'evaluation'
    protocol = dict(split=manifest['split_sha256'], config={k: cfg[k] for k in ('evaluation', 'imgsz', 'smoke')},
                    implementation_sha256=code_signature(),
                    checkpoints={k: file_hash(v) for k, v in checkpoints.items()},
                    selection='source-val fitness; fixed final adaptation epoch; O2O Student only')
    if not check_stage(directory, protocol, resume):
        return
    outputs = []
    for stage, checkpoint in checkpoints.items():
        model = YOLO(str(checkpoint), task='segment')
        if model.model.model[-1].nc != 4 or not model.model.end2end:
            raise ValueError('Evaluation requires a native end-to-end four-class Segment26')
        if stage == 'source_only':
            name = 'source_val.json'
            evaluate(model, dst, manifest, source, 'source_val', cfg, directory / name, checkpoint)
            outputs.append(name)
        name = f'{stage}_target_test.json'
        evaluate(model, dst, manifest, target, 'target_test', cfg, directory / name, checkpoint)
        outputs.append(name)
    mark_complete(directory, outputs)


def summarize(project, directions):
    rows = []
    for direction in directions:
        for stage in ('source_only', 'adabn', 'medrt_sfseg'):
            path = project / direction / 'evaluation' / f'{stage}_target_test.json'
            if not path.exists() or not (path.parent / 'complete.json').exists():
                rows.append(dict(direction=direction, stage=stage, region=None, status='not_evaluated'))
                continue
            report = json.loads(path.read_text())
            for region, stats in report['summary'].items():
                rows.append(dict(direction=direction, stage=stage, region=region, status='smoke' if report['smoke'] else 'complete',
                                 dice=stats['dice']['case_mean'], dice_patient=stats['dice']['patient_mean'],
                                 dice_gt_present=stats['dice_gt_present_mean'], asd_mm=stats['asd_mm']['case_mean'],
                                 hd95_mm=stats['hd95_mm']['case_mean'], undefined_surface_cases=stats['undefined_surface_cases']))
    write_json(project / 'summary.json', rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (project / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(f'Summary: {project / "summary.csv"}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=REPO / 'configs/experiments/brats2024_sfseg.yaml')
    p.add_argument('--stage', choices=('plan', 'prepare', 'source', 'adapt', 'evaluate', 'predict', 'summarize', 'all'), default='plan')
    p.add_argument('--directions', nargs='+', choices=DIRECTIONS)
    for key in ('raw-root', 'prepared-root', 'project', 'device', 'weights'):
        p.add_argument('--' + key)
    for key in ('batch', 'workers', 'imgsz', 'seed', 'source-epochs', 'adapt-epochs'):
        p.add_argument('--' + key, type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args(); cfg = effective_config(args)
    directions = args.directions or cfg['directions']
    if not directions or len(set(directions)) != len(directions) or not set(directions) <= set(DIRECTIONS):
        raise ValueError('Choose unique supported directions')
    suffix = 'smoke' if cfg['smoke'] else 'full'
    root = resolve_path(cfg['raw_root']); dst = resolve_path(cfg['prepared_root']) / suffix
    project = resolve_path(cfg['project']) / suffix
    project.mkdir(parents=True, exist_ok=True)
    if args.stage == 'summarize':
        summarize(project, directions); return
    if args.stage in ('plan', 'prepare', 'all'):
        plan = make_plan(root, cfg, args.smoke)
        plan_path = project / 'plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text())['split_sha256'] != plan['split_sha256']:
            raise ValueError('Existing project has a different split. Choose a new --project.')
        write_json(plan_path, plan)
        print(json.dumps(plan['full_counts'], indent=2), flush=True)
        if args.stage == 'plan':
            print(f'Plan: {plan_path}'); return
        manifest = prepare(root, dst, plan)
        if args.stage == 'prepare':
            return
    else:
        manifest = json.loads((dst / 'manifest.json').read_text())
        verify_prepared(dst, manifest)
        if (manifest['seed'] != cfg['seed'] or manifest['fractions'] != cfg['split']['fractions']
                or manifest['smoke'] != cfg['smoke']):
            raise ValueError('Configuration and prepared split disagree')
    for direction in directions:
        source, target = direction.split('_to_')
        run = project / direction; run.mkdir(parents=True, exist_ok=True)
        if args.stage in ('source', 'all'):
            train_source(dst, manifest, source, run / 'source', cfg, args.resume)
        if args.stage in ('adapt', 'all'):
            run_adaptation(dst, manifest, target, run, cfg, args.resume)
        if args.stage in ('evaluate', 'all'):
            run_evaluation(dst, manifest, source, target, run, cfg, args.resume)
        if args.stage == 'predict':
            from ultralytics import YOLO
            from brats2024_eval import predict_external
            if not (run / 'adapt' / 'complete.json').exists():
                raise ValueError('Complete adaptation before external prediction')
            predict_external(YOLO(str(run / 'adapt/student_final.pt')), dst, manifest, target, cfg,
                             run / 'external_predictions')
    summarize(project, directions)


if __name__ == '__main__':
    main()
