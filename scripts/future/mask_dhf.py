"""Mask-aware Dual-Head Fusion utilities for MedRT-SFOD.

This file is the proposed Try-3 extension of RT-SFOD's box-only DHF.  It is
kept independent from Ultralytics internals so it can be unit-tested before the
full segmentation training loop is wired.

Recommended paper ablation modes:
  - ``box``    : original DHF behaviour (control)
  - ``mask``   : redundancy determined by pseudo-mask IoU
  - ``hybrid`` : alpha * box-IoU + (1-alpha) * mask-IoU

Pseudo-mask reliability is estimated without ground truth using threshold
stability: a confident mask should change little when binarized at 0.4 vs 0.6.
This is a *proposed extension*, not something claimed by the original RT-SFOD
paper, and should be validated by ablation before becoming part of the method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

FusionMode = Literal["box", "mask", "hybrid"]


@dataclass
class InstancePredictions:
    boxes: torch.Tensor   # [N,4] xyxy, pixels
    scores: torch.Tensor  # [N]
    classes: torch.Tensor # [N]
    masks: torch.Tensor   # [N,H,W], probabilities in [0,1]
    quality: Optional[torch.Tensor] = None  # [N], mask reliability

    @property
    def device(self):
        return self.boxes.device

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def subset(self, idx: torch.Tensor) -> "InstancePredictions":
        q = None if self.quality is None else self.quality[idx]
        return InstancePredictions(self.boxes[idx], self.scores[idx], self.classes[idx], self.masks[idx], q)

    @staticmethod
    def empty(device: torch.device, hw: tuple[int, int], dtype: torch.dtype = torch.float32) -> "InstancePredictions":
        h, w = hw
        return InstancePredictions(
            boxes=torch.zeros((0, 4), device=device, dtype=dtype),
            scores=torch.zeros((0,), device=device, dtype=dtype),
            classes=torch.zeros((0,), device=device, dtype=torch.long),
            masks=torch.zeros((0, h, w), device=device, dtype=dtype),
            quality=torch.zeros((0,), device=device, dtype=dtype),
        )


def concat_predictions(a: InstancePredictions, b: InstancePredictions) -> InstancePredictions:
    if len(a) == 0:
        return b
    if len(b) == 0:
        return a
    qa = a.quality if a.quality is not None else torch.ones_like(a.scores)
    qb = b.quality if b.quality is not None else torch.ones_like(b.scores)
    return InstancePredictions(
        boxes=torch.cat([a.boxes, b.boxes], dim=0),
        scores=torch.cat([a.scores, b.scores], dim=0),
        classes=torch.cat([a.classes, b.classes], dim=0),
        masks=torch.cat([a.masks, b.masks], dim=0),
        quality=torch.cat([qa, qb], dim=0),
    )


def pairwise_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    a1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    a2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    return inter / (a1 + a2 - inter + eps)


def pairwise_mask_iou(masks1: torch.Tensor, masks2: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    if masks1.numel() == 0 or masks2.numel() == 0:
        return masks1.new_zeros((masks1.shape[0], masks2.shape[0]))
    a = (masks1 >= threshold).flatten(1).float()
    b = (masks2 >= threshold).flatten(1).float()
    inter = a @ b.T
    area_a = a.sum(dim=1, keepdim=True)
    area_b = b.sum(dim=1, keepdim=True).T
    return inter / (area_a + area_b - inter + eps)


def mask_stability_quality(masks: torch.Tensor, low: float = 0.4, high: float = 0.6, eps: float = 1e-7) -> torch.Tensor:
    """IoU between low/high-threshold masks; high means a stable pseudo-mask."""
    if masks.numel() == 0:
        return masks.new_zeros((masks.shape[0],))
    lo = (masks >= low).flatten(1)
    hi = (masks >= high).flatten(1)
    inter = (lo & hi).sum(dim=1).float()
    union = (lo | hi).sum(dim=1).float()
    # If both are empty, quality should be zero rather than spuriously perfect.
    return torch.where(union > 0, inter / (union + eps), torch.zeros_like(union))


def combined_reliability(preds: InstancePredictions) -> torch.Tensor:
    q = preds.quality if preds.quality is not None else mask_stability_quality(preds.masks)
    return preds.scores.clamp(0, 1) * q.clamp(0, 1)


def _overlap_matrix(a: InstancePredictions, b: InstancePredictions, mode: FusionMode, hybrid_alpha: float) -> torch.Tensor:
    if mode == "box":
        return pairwise_box_iou(a.boxes, b.boxes)
    if mode == "mask":
        return pairwise_mask_iou(a.masks, b.masks)
    if mode != "hybrid":
        raise ValueError(f"Unknown fusion mode: {mode}")
    alpha = float(np.clip(hybrid_alpha, 0.0, 1.0))
    return alpha * pairwise_box_iou(a.boxes, b.boxes) + (1.0 - alpha) * pairwise_mask_iou(a.masks, b.masks)


def instance_nms(preds: InstancePredictions, iou_threshold: float, mode: FusionMode = "mask", hybrid_alpha: float = 0.5) -> InstancePredictions:
    """Greedy class-wise NMS using box/mask/hybrid overlap."""
    if len(preds) <= 1:
        return preds
    keep_all: list[int] = []
    reliability = combined_reliability(preds)
    for cls in preds.classes.long().unique(sorted=True):
        cls_idx = torch.where(preds.classes.long() == cls)[0]
        order = cls_idx[torch.argsort(reliability[cls_idx], descending=True)]
        while order.numel():
            i = int(order[0].item())
            keep_all.append(i)
            if order.numel() == 1:
                break
            rest = order[1:]
            ov = _overlap_matrix(preds.subset(torch.tensor([i], device=preds.device)), preds.subset(rest), mode, hybrid_alpha)[0]
            order = rest[ov <= iou_threshold]
    keep = torch.tensor(keep_all, device=preds.device, dtype=torch.long)
    return preds.subset(keep)


def _filter_predictions(preds: InstancePredictions, score_thr: float, min_mask_quality: float) -> InstancePredictions:
    q = preds.quality if preds.quality is not None else mask_stability_quality(preds.masks)
    preds = InstancePredictions(preds.boxes, preds.scores, preds.classes, preds.masks, q)
    keep = (preds.scores >= score_thr) & (q >= min_mask_quality)
    return preds.subset(torch.where(keep)[0])


def mask_aware_dual_head_fusion(
    one2one: InstancePredictions,
    one2many: InstancePredictions,
    tau_o2o: float = 0.5,
    tau_o2m: float = 0.5,
    tau_no: float = 0.2,
    tau_dup: float = 0.7,
    mode: FusionMode = "hybrid",
    hybrid_alpha: float = 0.5,
    min_mask_quality: float = 0.0,
) -> InstancePredictions:
    """O2O anchors + non-redundant O2M extras, extended to pseudo-masks.

    ``mode='box'`` recovers the conceptual original DHF selection rule (apart
    from optional mask-quality filtering).  ``mask`` and ``hybrid`` are the
    proposed medical instance-segmentation variants to compare in ablation.
    """
    anchors = _filter_predictions(one2one, tau_o2o, min_mask_quality)
    candidates = _filter_predictions(one2many, tau_o2m, min_mask_quality)

    if len(candidates) == 0:
        fused = anchors
    elif len(anchors) == 0:
        fused = instance_nms(candidates, tau_dup, mode=mode, hybrid_alpha=hybrid_alpha)
    else:
        overlap = _overlap_matrix(candidates, anchors, mode, hybrid_alpha)
        extras_idx = torch.where(overlap.max(dim=1).values <= tau_no)[0]
        extras = candidates.subset(extras_idx)
        extras = instance_nms(extras, tau_dup, mode=mode, hybrid_alpha=hybrid_alpha)
        fused = concat_predictions(anchors, extras)

    if len(fused):
        order = torch.argsort(combined_reliability(fused), descending=True)
        fused = fused.subset(order)
    return fused


# -----------------------------------------------------------------------------
# Prototype-mask reconstruction and weak -> strong mapping
# -----------------------------------------------------------------------------


def crop_masks_to_boxes(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Zero mask probabilities outside each xyxy box (same image coordinates)."""
    if masks.numel() == 0:
        return masks
    n, h, w = masks.shape
    yy = torch.arange(h, device=masks.device)[None, :, None]
    xx = torch.arange(w, device=masks.device)[None, None, :]
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    inside = (
        (xx >= x1[:, None, None]) & (xx < x2[:, None, None]) &
        (yy >= y1[:, None, None]) & (yy < y2[:, None, None])
    )
    return masks * inside.to(masks.dtype)


def reconstruct_proto_masks(
    coeffs: torch.Tensor,
    proto: torch.Tensor,
    output_hw: tuple[int, int],
    boxes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reconstruct per-instance probability masks from prototype coefficients.

    ``proto`` is ``[C,Hm,Wm]`` for one image and ``coeffs`` is ``[N,C]``.
    """
    if coeffs.numel() == 0:
        h, w = output_hw
        return proto.new_zeros((0, h, w))
    c, hm, wm = proto.shape
    if coeffs.shape[1] != c:
        raise ValueError(f"Mask coefficient/prototype channel mismatch: {coeffs.shape[1]} vs {c}")
    logits = coeffs @ proto.reshape(c, -1)
    masks = logits.sigmoid().reshape(-1, hm, wm)
    masks = F.interpolate(masks[:, None], size=output_hw, mode="bilinear", align_corners=False)[:, 0]
    if boxes is not None:
        masks = crop_masks_to_boxes(masks, boxes)
    return masks


@torch.no_grad()
def warp_masks_weak_to_strong(
    masks: torch.Tensor,
    weak_info: dict,
    strong_info: dict,
    output_hw: tuple[int, int],
) -> torch.Tensor:
    """Apply the same strong-only affine/perspective geometry to pseudo-masks.

    The RT-SFOD dataloader shares the horizontal flip between weak and strong
    views, so no flip is applied here.  Masks are cropped to the valid resized
    image before warping, then placed in the top-left padded strong canvas.
    """
    if weak_info.get("flipped") != strong_info.get("flipped"):
        raise ValueError("Weak/strong flips must be shared before mask mapping.")
    n = masks.shape[0]
    out_h, out_w = output_hw
    if n == 0:
        return masks.new_zeros((0, out_h, out_w))

    valid_h, valid_w = strong_info["final_size"]
    src_h, src_w = weak_info["final_size"]
    if (src_h, src_w) != (valid_h, valid_w):
        raise ValueError("Current mapper assumes weak/strong share the same pre-geometry resize.")

    result = []
    for m in masks.detach().float().cpu().numpy():
        m = m[:src_h, :src_w]
        affine = strong_info.get("scale_translate_matrix")
        if affine is not None:
            m = cv2.warpAffine(m, affine, (valid_w, valid_h), flags=cv2.INTER_LINEAR, borderValue=0.0)
        persp = strong_info.get("perspective_matrix")
        if persp is not None:
            m = cv2.warpPerspective(m, persp, (valid_w, valid_h), flags=cv2.INTER_LINEAR, borderValue=0.0)
        canvas = np.zeros((out_h, out_w), dtype=np.float32)
        canvas[:valid_h, :valid_w] = np.clip(m, 0.0, 1.0)
        result.append(canvas)
    return torch.from_numpy(np.stack(result)).to(device=masks.device, dtype=masks.dtype)
