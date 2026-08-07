"""Milestone-2: source-free YOLO26 instance-segmentation adaptation.

Purpose: establish a CLEAN segmentation baseline before adding CARD-v2.
It reuses the original RT-SFOD weak/strong pipeline and MARD, but replaces the
box-only teacher pseudo labels with pseudo boxes + pseudo masks and supports
box/mask/hybrid DHF ablations via ``mask_dhf.py``.

Expected placement inside the project:
    scripts/YOLO26/stage2_medseg_rtsfod_yolo26.py
next to:
    stage2_rtsfod_yolo26.py
    mask_dhf.py

Compatibility assumption (checked against current YOLO26 Segment26 API):
- eval: ``((final_o2o, proto), branch_preds)``
- final predictions columns: xyxy, score, class, mask coefficients
- training branches expose ``one2one``/``one2many`` and ``proto``
- native SegmentationModel criterion accepts batch_idx/cls/bboxes/masks and,
  for Proto26 semantic auxiliary output, sem_masks.

Run this WITHOUT CARD first.  Once this baseline is stable, wire the same
pseudo-mask reliability score into CARD-v2 rather than debugging segmentation
and compression simultaneously.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.ops import xyxy2xywh  # noqa: E402

from mask_dhf import (  # noqa: E402
    InstancePredictions,
    combined_reliability,
    mask_aware_dual_head_fusion,
    mask_stability_quality,
    reconstruct_proto_masks,
    warp_masks_weak_to_strong,
)
from stage2_rtsfod_yolo26 import (  # noqa: E402
    MARD_ALPHA,
    MARD_BETA,
    MARD_BOX_CONF_THRESHOLD,
    MARD_ETA,
    MARD_FG_POINTS,
    MARD_GATE_THRESHOLD,
    MARD_GAMMA,
    MARD_INTERVAL,
    MARD_LAMBDA0,
    MARD_LAMBDA_MAX,
    MARD_TOPK_BOXES,
    MARD_WARMUP_EPOCHS,
    MARD_BG_POINTS,
    TargetTeacherStudentDataset,
    collate_fn,
    compute_mard_loss,
    list_images_from_yaml,
    mard_weight,
    resolve_device,
    scalarize,
    seed_everything,
    seed_worker,
    transform_boxes_weak_to_strong,
    update_teacher_ema,
    val_device_arg,
)

DEFAULT_IMGSZ = 640
DEFAULT_BATCH = 8
DEFAULT_EPOCHS = 60
DEFAULT_LR = 1e-4
EMA_MOMENTUM = 0.999


class PyramidInputFeatureHook:
    """Capture P3/P4/P5 passed into final Detect/Segment head."""
    def __init__(self, model: nn.Module):
        top = getattr(model, "model", None)
        if top is None or len(top) == 0:
            raise RuntimeError("Model has no top-level model sequence")
        self.head = top[-1]
        name = self.head.__class__.__name__.lower()
        if "segment" not in name and "detect" not in name:
            raise RuntimeError(f"Expected final Detect/Segment head, got {type(self.head)}")
        self.latest: Optional[list[torch.Tensor]] = None
        self.handle = self.head.register_forward_pre_hook(self._hook)

    def _hook(self, _module, inputs):
        x = inputs[0] if len(inputs) == 1 else inputs
        if isinstance(x, (list, tuple)) and len(x) >= 3 and all(torch.is_tensor(t) and t.ndim == 4 for t in x[:3]):
            self.latest = list(x[:3])
        else:
            self.latest = None

    def close(self):
        if self.handle is not None:
            self.handle.remove(); self.handle = None


def ensure_segment_model(model: nn.Module, label: str) -> None:
    if not getattr(model, "end2end", False):
        raise ValueError(f"{label} must be an end-to-end YOLO26 model")
    head = model.model[-1]
    if "segment" not in head.__class__.__name__.lower():
        raise ValueError(f"{label} must end in Segment/Segment26, got {type(head)}")
    if not hasattr(head, "one2one") or not hasattr(head, "one2many") or not hasattr(head, "nm"):
        raise ValueError(f"{label} segmentation head lacks dual-head/mask interface")


def ensure_segmentation_loss_args(model: nn.Module, epochs: int) -> None:
    existing = getattr(model, "args", None)
    if existing is None:
        existing = argparse.Namespace()
    elif isinstance(existing, dict):
        existing = argparse.Namespace(**existing)
    defaults = {
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "epochs": int(epochs),
        "overlap_mask": False,  # critical: our pseudo masks are per-instance
    }
    for key, value in defaults.items():
        if not hasattr(existing, key) or getattr(existing, key) is None:
            setattr(existing, key, value)
    # Force non-overlap representation for source-free pseudo instances.
    existing.overlap_mask = False
    model.args = existing


def setup_teacher_student(checkpoint: str, device: torch.device, epochs: int):
    tw = YOLO(checkpoint, task="segment")
    teacher = tw.model.to(device).float()
    ensure_segment_model(teacher, "Teacher")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    sw = YOLO(checkpoint, task="segment")
    student = sw.model.to(device).float()
    ensure_segment_model(student, "Student")
    for p in student.parameters():
        p.requires_grad_(True)
    ensure_segmentation_loss_args(student, epochs)
    student.criterion = student.init_criterion()
    return teacher, student, sw


def _proto_tensor(proto):
    """Proto26 may return (instance_proto, semantic_logits)."""
    if isinstance(proto, (tuple, list)):
        return proto[0]
    return proto


def _rows_to_instances(rows: torch.Tensor, proto_i: torch.Tensor, output_hw: tuple[int, int], nm: int) -> InstancePredictions:
    device = proto_i.device
    if rows is None or rows.numel() == 0:
        return InstancePredictions.empty(device, output_hw)
    rows = rows[rows[:, 4] > 0]
    if rows.numel() == 0:
        return InstancePredictions.empty(device, output_hw)
    if rows.shape[1] < 6 + nm:
        raise RuntimeError(f"Segment postprocess row has {rows.shape[1]} cols; expected >= {6+nm}")
    boxes = rows[:, :4]
    scores = rows[:, 4]
    classes = rows[:, 5].long()
    coeffs = rows[:, 6 : 6 + nm]
    masks = reconstruct_proto_masks(coeffs, proto_i, output_hw=output_hw, boxes=boxes)
    quality = mask_stability_quality(masks)
    return InstancePredictions(boxes, scores, classes, masks, quality)


@torch.no_grad()
def generate_pseudo_instances(teacher: nn.Module, weak_imgs: torch.Tensor, args) -> list[InstancePredictions]:
    teacher.eval()
    outputs = teacher(weak_imgs, augment=False, visualize=False)
    if not (isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], dict)):
        raise RuntimeError("Expected Segment26 eval output ((o2o, proto), branch_preds)")
    first, branch_preds = outputs
    if not (isinstance(first, tuple) and len(first) == 2):
        raise RuntimeError("Expected first Segment26 eval item to be (final_o2o, proto)")
    final_o2o, proto = first
    proto = _proto_tensor(proto)
    head = teacher.model[-1]
    nm = int(head.nm)

    decoded_o2m = head._inference(branch_preds["one2many"]).permute(0, 2, 1)
    final_o2m = head.postprocess(decoded_o2m)

    h, w = weak_imgs.shape[-2:]
    results = []
    for i in range(weak_imgs.shape[0]):
        o2o = _rows_to_instances(final_o2o[i], proto[i], (h, w), nm)
        o2m = _rows_to_instances(final_o2m[i], proto[i], (h, w), nm)
        fused = mask_aware_dual_head_fusion(
            o2o, o2m,
            tau_o2o=args.tau_o2o,
            tau_o2m=args.tau_o2m,
            tau_no=args.tau_no,
            tau_dup=args.tau_dup,
            mode=args.mask_dhf_mode,
            hybrid_alpha=args.mask_dhf_alpha,
            min_mask_quality=args.min_mask_quality,
        )
        results.append(fused)
    return results


@torch.no_grad()
def map_instances_to_strong(
    pseudo_weak: list[InstancePredictions],
    weak_infos: list[dict],
    strong_infos: list[dict],
    output_hw: tuple[int, int],
) -> list[InstancePredictions]:
    out = []
    for pred, wi, si in zip(pseudo_weak, weak_infos, strong_infos):
        if len(pred) == 0:
            out.append(InstancePredictions.empty(pred.device, output_hw))
            continue
        boxes, valid = transform_boxes_weak_to_strong(pred.boxes, wi, si)
        masks = warp_masks_weak_to_strong(pred.masks, wi, si, output_hw)
        idx = torch.where(valid)[0]
        if idx.numel() == 0:
            out.append(InstancePredictions.empty(pred.device, output_hw))
            continue
        mapped = InstancePredictions(
            boxes=boxes[idx],
            scores=pred.scores[idx],
            classes=pred.classes[idx],
            masks=masks[idx],
            quality=pred.quality[idx] if pred.quality is not None else None,
        )
        out.append(mapped)
    return out


def pseudo_reliability(preds: list[InstancePredictions]) -> float:
    vals = [combined_reliability(p) for p in preds if len(p)]
    return float(torch.cat(vals).mean().item()) if vals else 0.0


def to_box_labels(preds: list[InstancePredictions]) -> list[torch.Tensor]:
    labels = []
    for p in preds:
        if len(p) == 0:
            labels.append(p.boxes.new_zeros((0, 6)))
        else:
            labels.append(torch.cat([p.boxes, p.scores[:, None], p.classes.float()[:, None]], dim=1))
    return labels


def _semantic_index_map(preds: list[InstancePredictions], hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    """Per-image class index map for optional Proto26 semantic auxiliary loss."""
    h, w = hw
    sem = torch.zeros((len(preds), h, w), device=device, dtype=torch.long)
    for bi, p in enumerate(preds):
        if len(p) == 0:
            continue
        # High reliability first; later assignments only fill not-yet-covered pixels.
        order = torch.argsort(combined_reliability(p), descending=True)
        occupied = torch.zeros((h, w), device=device, dtype=torch.bool)
        for j in order:
            region = p.masks[j] >= 0.5
            fill = region & (~occupied)
            sem[bi][fill] = p.classes[j].long()
            occupied |= region
    return sem


def compute_student_seg_loss(student_outputs, pseudo: list[InstancePredictions], student: nn.Module, criterion, input_shape):
    if not isinstance(student_outputs, dict) or "one2one" not in student_outputs or "one2many" not in student_outputs:
        raise RuntimeError("Segment student train output must expose one2one/one2many branches")
    branch = student_outputs["one2many"]
    proto = _proto_tensor(branch["proto"])
    device = proto.device
    _, _, mh, mw = proto.shape
    in_h, in_w = int(input_shape[2]), int(input_shape[3])
    norm = torch.tensor([in_w, in_h, in_w, in_h], device=device, dtype=torch.float32)

    batch_idx, classes, boxes, masks = [], [], [], []
    for bi, p in enumerate(pseudo):
        if len(p) == 0:
            continue
        batch_idx.append(torch.full((len(p),), bi, device=device, dtype=torch.long))
        classes.append(p.classes.float())
        boxes.append(xyxy2xywh(p.boxes) / norm)
        # Native segmentation criterion accepts individual binary masks when overlap_mask=False.
        m = F.interpolate(p.masks[:, None].float(), size=(mh, mw), mode="bilinear", align_corners=False)[:, 0]
        masks.append((m >= 0.5).float())

    if not boxes:
        zero = sum((param.sum() * 0.0) for param in student.parameters())
        return zero, {}

    # For semantic auxiliary proto branch, a class-index map is also supplied.
    pseudo_resized = []
    cursor = 0
    for p in pseudo:
        if len(p) == 0:
            pseudo_resized.append(InstancePredictions.empty(device, (mh, mw)))
            continue
        m = F.interpolate(p.masks[:, None].float(), size=(mh, mw), mode="bilinear", align_corners=False)[:, 0]
        scale = torch.tensor([mw / in_w, mh / in_h, mw / in_w, mh / in_h], device=device)
        pseudo_resized.append(InstancePredictions(p.boxes * scale, p.scores, p.classes, m, p.quality))

    batch = {
        "batch_idx": torch.cat(batch_idx),
        "cls": torch.cat(classes),
        "bboxes": torch.cat(boxes),
        "masks": torch.cat(masks),
        "sem_masks": _semantic_index_map(pseudo_resized, (mh, mw), device),
    }
    total, items = criterion(student_outputs, batch)
    total = total.sum() if torch.is_tensor(total) else total
    return total, items


def parse_loss_items(items) -> str:
    if isinstance(items, dict):
        parts = []
        for k, v in items.items():
            try: parts.append(f"{k}={scalarize(v):.4f}")
            except Exception: pass
        return " ".join(parts)
    if torch.is_tensor(items):
        return "loss_items=" + ",".join(f"{float(x):.4f}" for x in items.detach().flatten())
    return ""


def main(args):
    seed_everything(args.seed, args.deterministic)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)

    imgs = list_images_from_yaml(args.data)
    if not imgs:
        raise RuntimeError(f"No target images in {args.data}")
    ds = TargetTeacherStudentDataset(imgs, img_size=args.imgsz)
    gen = torch.Generator(); gen.manual_seed(args.seed if args.seed >= 0 else 0)
    loader = data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True,
                             collate_fn=collate_fn, drop_last=False, worker_init_fn=seed_worker,
                             generator=gen, persistent_workers=args.workers > 0)

    teacher, student, student_wrapper = setup_teacher_student(args.stage1_model, device, args.epochs)
    criterion = student.criterion
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=0.937, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    hook = PyramidInputFeatureHook(student)
    global_step = 0

    try:
        for epoch in range(args.epochs):
            student.train(); t0 = time.time(); valid_batches = 0; skipped = 0
            for bi, (weak, strong, _paths, weak_infos, strong_infos) in enumerate(loader, start=1):
                weak = weak.to(device, non_blocking=True); strong = strong.to(device, non_blocking=True)
                pw = generate_pseudo_instances(teacher, weak, args)
                ps = map_instances_to_strong(pw, weak_infos, strong_infos, output_hw=tuple(strong.shape[-2:]))
                valid = [len(p) > 0 for p in ps]
                if not any(valid):
                    skipped += 1; global_step += 1; continue

                strong_v = strong[valid]
                p_v = [p for p, keep in zip(ps, valid) if keep]
                info_v = [inf for inf, keep in zip(strong_infos, valid) if keep]
                reliability = pseudo_reliability(p_v)

                hook.latest = None
                outputs = student(strong_v)
                feats = hook.latest
                detseg_loss, loss_items = compute_student_seg_loss(outputs, p_v, student, criterion, strong_v.shape)

                labels_v = to_box_labels(p_v)
                mard = detseg_loss.new_zeros(())
                lm = 0.0
                if feats is not None and global_step % MARD_INTERVAL == 0:
                    mard, _ = compute_mard_loss(feats, labels_v, info_v, int(strong_v.shape[2]), int(strong_v.shape[3]), args)
                    # Use box*mask reliability rather than box confidence only for segmentation.
                    lm = mard_weight(args, global_step, len(loader), reliability)

                total = detseg_loss + lm * mard
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                optimizer.step()
                global_step += 1; valid_batches += 1

                if args.print_freq > 0 and (bi == 1 or bi % args.print_freq == 0 or bi == len(loader)):
                    ninst = sum(len(p) for p in p_v)
                    print(
                        f"[E{epoch+1:03d} B{bi:04d}/{len(loader):04d}] total={scalarize(total):.4f} "
                        f"segdet={scalarize(detseg_loss):.4f} mard={scalarize(mard):.4f} lambda={lm:.4f} "
                        f"instances={ninst} reliability={reliability:.4f} {parse_loss_items(loss_items)}",
                        flush=True,
                    )

            scheduler.step()
            if hasattr(criterion, "update"):
                criterion.update()
            update_teacher_ema(teacher, student, args.ema_momentum)
            print(f"[Epoch {epoch+1}] time={time.time()-t0:.1f}s valid={valid_batches} skipped={skipped}")

            if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
                path = ckpt_dir / f"yolo26_medseg_sfod_epoch_{epoch+1}.pt"
                student_wrapper.model = student; student_wrapper.save(str(path)); print(f"saved {path}")

            if args.eval and (epoch + 1) % args.val_interval == 0:
                student.eval(); vw = YOLO(args.stage1_model, task="segment"); vw.model = student
                metrics = vw.val(data=args.data, imgsz=args.imgsz, batch=args.batch, device=val_device_arg(device), verbose=False, plots=False)
                seg = getattr(metrics, "seg", None)
                print(f"[ORACLE eval only] mask mAP50={getattr(seg, 'map50', 0.0):.4f}")
                student.train()
    finally:
        hook.close()


def parse_args():
    ap = argparse.ArgumentParser(description="MedRT-SFOD Stage2 instance-seg baseline (Mask-DHF + MARD, no CARD)")
    ap.add_argument("--stage1_model", required=True, help="Source-seg checkpoint after target AdaBN")
    ap.add_argument("--data", required=True, help="UNLABELED target data YAML for adaptation; labels are only touched if --eval")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--grad_clip", type=float, default=10.0)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tau_o2o", type=float, default=0.5)
    ap.add_argument("--tau_o2m", type=float, default=0.5)
    ap.add_argument("--tau_no", type=float, default=0.2)
    ap.add_argument("--tau_dup", type=float, default=0.7)
    ap.add_argument("--mask_dhf_mode", choices=["box", "mask", "hybrid"], default="hybrid")
    ap.add_argument("--mask_dhf_alpha", type=float, default=0.5)
    ap.add_argument("--min_mask_quality", type=float, default=0.0, help="Start at 0 for baseline; ablate >0 only after calibration")

    ap.add_argument("--mard_lambda0", type=float, default=MARD_LAMBDA0)
    ap.add_argument("--mard_lambda_max", type=float, default=MARD_LAMBDA_MAX)
    ap.add_argument("--mard_gamma", type=float, default=MARD_GAMMA)
    ap.add_argument("--mard_alpha", type=float, default=MARD_ALPHA)
    ap.add_argument("--mard_beta", type=float, default=MARD_BETA)
    ap.add_argument("--mard_warmup_epochs", type=float, default=MARD_WARMUP_EPOCHS)
    ap.add_argument("--mard_gate_threshold", type=float, default=MARD_GATE_THRESHOLD)
    ap.add_argument("--mard_topk_boxes", type=int, default=MARD_TOPK_BOXES)
    ap.add_argument("--mard_fg_points", type=int, default=MARD_FG_POINTS)
    ap.add_argument("--mard_bg_points", type=int, default=MARD_BG_POINTS)
    ap.add_argument("--mard_eta", type=float, default=MARD_ETA)
    ap.add_argument("--ema_momentum", type=float, default=EMA_MOMENTUM)
    ap.add_argument("--print_freq", type=int, default=10)
    ap.add_argument("--save_interval", type=int, default=1)
    ap.add_argument("--eval", action="store_true", help="Target-label evaluation only; never use for adaptation/model selection")
    ap.add_argument("--val_interval", type=int, default=1)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--deterministic", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
