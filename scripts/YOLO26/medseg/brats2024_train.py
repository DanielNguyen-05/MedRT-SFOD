"""Image-only AdaBN / Mean Teacher + class-wise DURR + SegMARD-v2.

This entry point accepts no source image, target mask, or test dataset argument.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from ultralytics import YOLO

from brats2024_data import write_json
from durr_multiclass import (decode_teacher, generate_multiclass_pseudo,
                             build_multiclass_pseudo_batch, multiclass_durr_losses, select_batch)
from durr_seg import _same_class_best_iou, durr_ramp
from mask_dhf_seg import mask_probs_from_coefficients, mask_stability, classwise_nms_indices
from segmard_seg import compute_segmard_loss
from smoke_stage2_mt_seg import (TargetMTDataset, collate, list_images, seed_everything,
                                 setup_teacher_student, update_teacher_ema)
from stage1_adabn_seg import (TargetImageDataset, collate_target, freeze_except_bn_stats,
                              capture_bn_stats, compute_bn_delta)
from stage2_dense_sfseg_durr import SegmentInputFeatureHook, average_confidence, mard_weight


def device_for(value):
    if value == 'cpu':
        return torch.device('cpu')
    if value == 'mps':
        if not torch.backends.mps.is_available():
            raise RuntimeError('MPS is unavailable in this environment; use CPU or run on a GPU host')
        return torch.device('mps')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; explicitly select --device cpu or mps')
    return torch.device(f'cuda:{value}' if str(value).isdigit() else value)


class MRITargetDataset(TargetMTDataset):
    @staticmethod
    def strong_photo(im):
        # One MRI modality replicated into RGB; never introduce hue/saturation.
        gray = im[..., 0].astype(np.float32) / 255
        foreground = gray > 0
        gray = np.clip(gray * random.uniform(0.85, 1.15), 0, 1) ** random.uniform(0.8, 1.2)
        if random.random() < 0.25:
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if random.random() < 0.5:
            gray += np.random.normal(0, 0.015, gray.shape).astype(np.float32)
        gray[~foreground] = 0
        gray = np.rint(np.clip(gray, 0, 1) * 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)


def adabn(weights, images, out, cfg, device):
    wrapper = YOLO(str(weights))
    net = wrapper.model.to(device).float()
    if freeze_except_bn_stats(net) == 0:
        raise ValueError('AdaBN requires an unfused checkpoint with BatchNorm')
    before = capture_bn_stats(net)
    loader = DataLoader(TargetImageDataset(images, cfg['imgsz'], flip_prob=0),
                        batch_size=cfg['batch'], shuffle=False, num_workers=cfg['workers'],
                        collate_fn=collate_target)
    with torch.inference_mode():
        for epoch in range(cfg['adabn_epochs']):
            for i, x in enumerate(loader):
                net(x.to(device))
                if i % 100 == 0:
                    print(f'AdaBN epoch {epoch + 1}: {i + 1}/{len(loader)}', flush=True)
    delta = compute_bn_delta(before, capture_bn_stats(net))
    if not delta['changed_bn_layers']:
        raise RuntimeError('No BN statistics changed')
    net.eval()
    wrapper.save(str(out / 'adabn.pt'))
    write_json(out / 'adabn.json', dict(delta, image_count=len(images), target_labels_used=False))


@torch.no_grad()
def calibrate(teacher, images, cfg, device):
    # Deterministic, image-only Q25 audit before any student optimization.
    limit = cfg['adaptation']['calibration_images']
    if limit and len(images) > limit:
        indices = np.linspace(0, len(images) - 1, limit).round().astype(int)
        selected = [images[i] for i in indices]
    else:
        selected = images
    loader = DataLoader(TargetImageDataset(selected, cfg['imgsz'], flip_prob=0),
                        batch_size=cfg['batch'], shuffle=False, num_workers=cfg['workers'],
                        collate_fn=collate_target)
    a = cfg['adaptation']; values = [[] for _ in range(4)]
    for batch_index, x in enumerate(loader):
        x = x.to(device)
        oo, om, proto = decode_teacher(teacher, x)
        for i in range(len(x)):
            anchors = oo[i][oo[i][:, 4] >= a['tau_o2o']]
            candidates = om[i][om[i][:, 4] >= a['tau_o2m']]
            ap = mask_probs_from_coefficients(anchors, proto[i], x.shape[2], x.shape[3])
            anchors = anchors[(ap >= a['mask_threshold']).sum((1, 2)) >= a['min_mask_pixels']]
            cp = mask_probs_from_coefficients(candidates, proto[i], x.shape[2], x.shape[3])
            keep = (cp >= a['mask_threshold']).sum((1, 2)) >= a['min_mask_pixels']
            candidates, cp = candidates[keep], cp[keep]
            best, _ = _same_class_best_iou(candidates, anchors)
            candidates, cp = candidates[best <= a['tau_no']], cp[best <= a['tau_no']]
            keep = classwise_nms_indices(candidates, a['tau_dup'])
            candidates, cp = candidates[keep], cp[keep]
            rel = (candidates[:, 4] * mask_stability(cp, a['stability_low'], a['stability_high'])).sqrt()
            for cls in range(4):
                values[cls].extend(rel[candidates[:, 5].long() == cls].cpu().tolist())
        if batch_index % 100 == 0:
            print(f'Label-free Q25 audit {batch_index + 1}/{len(loader)}', flush=True)
    # No candidates -> disable that class's O2M extras, retain any O2O anchors.
    thresholds = [float(np.quantile(v, .25)) if v else 1.000001 for v in values]
    return dict(thresholds=thresholds, candidate_counts=[len(v) for v in values], images=len(selected),
                definition='per-class Q25 of valid novel O2M candidates after class-wise NMS',
                empty_class_policy='threshold > 1 disables extras; no target labels consulted',
                target_labels_used=False)


def atomic_torch_save(value, path):
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def adapt(weights, images, out, cfg, device, resume=False):
    a = cfg['adaptation']; epochs = a['epochs']
    teacher, student, wrapper, criterion = setup_teacher_student(str(weights), device, epochs)
    if student.model[-1].nc != 4:
        raise ValueError('BraTS2024 requires a four-class source checkpoint')
    teacher_wrapper = YOLO(str(weights))
    optimizer = torch.optim.SGD(student.parameters(), lr=a['lr'], momentum=.937,
                                weight_decay=.0005, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=a['lr'] * .01)
    generator = torch.Generator().manual_seed(cfg['seed'])
    loader = DataLoader(MRITargetDataset(images, cfg['imgsz']), batch_size=cfg['batch'],
                        shuffle=True, num_workers=cfg['workers'], collate_fn=collate,
                        generator=generator, pin_memory=device.type == 'cuda')
    # SegMARD-v2 settings explicitly supplied, not inherited from polyp CLI defaults.
    mard = argparse.Namespace(**a['segmard'])
    start_epoch = global_step = total_updates = 0
    history = []
    if resume:
        state = torch.load(out / 'resume.pt', map_location=device, weights_only=False)
        student.load_state_dict(state['student']); teacher.load_state_dict(state['teacher'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_epoch, global_step, total_updates = state['epoch'], state['global_step'], state['total_updates']
        history = state['history']; calibration = state['calibration']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng'].cpu()); generator.set_state(state['loader_rng'].cpu())
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        if device.type == 'mps' and state.get('mps_rng') is not None:
            torch.mps.set_rng_state(state['mps_rng'].cpu())
        for _ in range(start_epoch):
            if hasattr(criterion, 'update'):
                criterion.update()
    else:
        calibration = calibrate(teacher, images, cfg, device)
        write_json(out / 'calibration.json', calibration)
    pseudo_kwargs = {k: a[k] for k in (
        'tau_o2o', 'tau_o2m', 'tau_no', 'tau_dup', 'tau_match', 'max_witnesses',
        'mask_threshold', 'stability_low', 'stability_high', 'min_mask_pixels',
        'boundary_kernel', 'route_gain', 'route_min_disagreement', 'rescue_conf',
        'rescue_stability', 'rescue_consensus_iou', 'rescue_min_support', 'evidence_conf',
        'safe_bg_teacher_prob', 'signed_mode')}
    hook = SegmentInputFeatureHook(student)
    try:
        for epoch in range(start_epoch, epochs):
            student.train(); updates = skipped = 0
            stats = dict(loss=0., sfseg=0., mard=0., directional=0., rescue=0., hallucination=0.,
                         pseudo_instances=0, routed_pixels=0., rescued_pixels=0., hallucination_pixels=0.)
            started = time.monotonic()
            for batch_i, (weak, strong, _) in enumerate(loader):
                weak, strong = weak.to(device), strong.to(device)
                labels, masks, _, pseudo_stats, routes = generate_multiclass_pseudo(
                    teacher, weak, 4, calibration['thresholds'], **pseudo_kwargs)
                optimizer.zero_grad(set_to_none=True)
                outputs = student(strong)
                feats = hook.latest
                batch = build_multiclass_pseudo_batch(labels, masks, strong.shape, 4)
                zero = next(student.parameters()).sum() * 0
                sfseg = mard_loss = zero
                if batch is not None:
                    # The native semantic loss's non-overlap path assumes every
                    # sample has instances. Restrict it to nonempty images.
                    valid = [i for i, rows in enumerate(labels) if len(rows)]
                    if len(valid) != len(labels):
                        valid_outputs = select_batch(outputs, valid)
                        feats = [f[valid] for f in feats]
                        valid_labels, valid_masks = [labels[i] for i in valid], [masks[i] for i in valid]
                        batch = build_multiclass_pseudo_batch(valid_labels, valid_masks, strong[valid].shape, 4)
                    else:
                        valid_outputs, valid_labels, valid_masks = outputs, labels, masks
                    sfseg = criterion(valid_outputs, batch)[0].sum()
                    mard_loss, _ = compute_segmard_loss(feats, valid_labels, valid_masks,
                                                       strong.shape[2], strong.shape[3], mard)
                dl, rl, hl, routing_stats = multiclass_durr_losses(
                    outputs, routes, student_threshold=a['hall_student_threshold'],
                    area_threshold=a['hall_area_threshold'], area_weight=a['hall_area_weight'])
                lm = mard_weight(mard, global_step, len(loader), average_confidence(labels))
                ld = durr_ramp(global_step, len(loader), a['warmup_epochs'])
                loss = sfseg + lm * mard_loss + ld * (a['lambda_dir'] * dl + a['lambda_rescue'] * rl + a['lambda_hall'] * hl)
                has_signal = batch is not None or routing_stats.get('hall_triggered_images', 0) > 0
                global_step += 1
                if not has_signal:
                    skipped += 1
                    continue
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite adaptation loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 10.)
                if not torch.isfinite(norm):
                    raise RuntimeError('Non-finite adaptation gradient')
                optimizer.step(); updates += 1; total_updates += 1
                for k, value in zip(('loss', 'sfseg', 'mard', 'directional', 'rescue', 'hallucination'),
                                    (loss, sfseg, mard_loss, dl, rl, hl)):
                    stats[k] += float(value.detach())
                stats['pseudo_instances'] += int(pseudo_stats['pseudo'])
                stats['routed_pixels'] += routing_stats.get('dir_pixels', 0)
                stats['rescued_pixels'] += routing_stats.get('rescue_pixels', 0)
                stats['hallucination_pixels'] += routing_stats.get('hall_pixels', 0)
                if batch_i % 50 == 0:
                    print(f'Adapt {epoch + 1}/{epochs} batch {batch_i + 1}/{len(loader)} loss={float(loss.detach()):.4f}', flush=True)
            if not updates:
                raise RuntimeError('No optimization signal in target images. Inspect source quality and calibration; target labels were not used.')
            scheduler.step()
            if hasattr(criterion, 'update'):
                criterion.update()
            # Preserve the existing method: epoch-level parameter EMA; Teacher BN
            # buffers remain those obtained by AdaBN (not Student running buffers).
            update_teacher_ema(teacher, student, a['ema'])
            history.append(dict(epoch=epoch + 1, updates=updates, skipped=skipped,
                                seconds=time.monotonic() - started, **stats))
            state = dict(epoch=epoch + 1, global_step=global_step, total_updates=total_updates,
                         student=student.state_dict(), teacher=teacher.state_dict(), optimizer=optimizer.state_dict(),
                         scheduler=scheduler.state_dict(), calibration=calibration, history=history,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                         loader_rng=generator.get_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                         mps_rng=torch.mps.get_rng_state() if device.type == 'mps' else None)
            atomic_torch_save(state, out / 'resume.pt')
            write_json(out / 'history.json', history)
            print(f'Adapt epoch {epoch + 1}: {updates} updates; checkpoint saved', flush=True)
    finally:
        hook.close()
    student.eval(); teacher.eval()
    wrapper.model = student; wrapper.save(str(out / 'student_final.pt'))
    teacher_wrapper.model = teacher; teacher_wrapper.save(str(out / 'teacher_final.pt'))
    write_json(out / 'adaptation.json', dict(method='MedRT-SFSeg: AdaBN + MT + class-wise Mask-DHF/DURR + SegMARD-v2',
                                           epochs=epochs, optimizer_steps=total_updates, target_labels_used=False,
                                           checkpoint_selection='fixed final epoch Student; no target GT selection',
                                           teacher_bn_policy='fixed AdaBN buffers; epoch-level parameter EMA',
                                           calibration=calibration, smoke=cfg['smoke']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('adabn', 'adapt'), required=True)
    p.add_argument('--weights', required=True)
    p.add_argument('--target-images', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seed_everything(cfg['seed'])
    device = device_for(cfg['device'])
    images = list_images(args.target_images)
    # Prevent invocation on a parent directory containing val/test/source images.
    if args.target_images.name != 'target_train' or any(x.parent != args.target_images for x in images):
        raise ValueError('Pass the flat target_train image directory only')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == 'adabn':
        adabn(args.weights, images, args.out_dir, cfg, device)
    else:
        adapt(args.weights, images, args.out_dir, cfg, device, args.resume)


if __name__ == '__main__':
    main()
