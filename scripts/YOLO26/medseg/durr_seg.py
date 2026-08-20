#!/usr/bin/env python3
"""
DURR: Dual-head Uncertainty Reliability Routing for MedRT-SFSeg.

DURR is a TRAINING-ONLY module built on top of the existing BDL/Mask-DHF
pseudo-label construction. It does not change deployment inference.

Core signals
------------
1) Reliable O2M Rescue:
   When native O2O is empty, a high-confidence, high-stability O2M prediction
   can act as a rescue supervision source. This is kept distinct from native
   O2O and is NOT used as a fake O2O witness for BDL.

2) Signed Boundary Routing:
   For matched native O2O/O2M predictions, use SIGNED disagreement
       Delta(x) = r_m P_m(x) - r_o P_o(x)
   on a two-sided boundary band. Positive Delta encourages expansion;
   negative Delta encourages shrinkage.

3) Safe Hallucination Suppression:
   If the final Teacher pseudo set is empty, Student foreground is penalized
   only in pixels where BOTH Teacher heads have low foreground evidence.
   Teacher silence is never treated as whole-image background ground truth.

No target GT is used anywhere in this file.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import box_iou

from boundary_dhf_seg import generate_boundary_dhf_pseudo_masks
from mask_dhf_seg import (
    classwise_nms_indices,
    mask_probs_from_coefficients,
    mask_stability,
)

EPS = 1e-8


def _same_class_best_iou(
    candidates: torch.Tensor,
    anchors: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidates.shape[0] == 0:
        return (
            candidates.new_zeros((0,)),
            torch.zeros((0,), device=candidates.device, dtype=torch.long),
        )
    if anchors.shape[0] == 0:
        return (
            candidates.new_full((candidates.shape[0],), -1.0),
            torch.full(
                (candidates.shape[0],),
                -1,
                device=candidates.device,
                dtype=torch.long,
            ),
        )

    ious = box_iou(candidates[:, :4], anchors[:, :4])
    same = (
        candidates[:, 5].long()[:, None]
        == anchors[:, 5].long()[None, :]
    )
    ious = torch.where(
        same,
        ious,
        torch.full_like(ious, -1.0),
    )
    return ious.max(dim=1)


def _binary_erode(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return mask.bool()
    if kernel % 2 == 0:
        raise ValueError("DURR boundary kernel must be odd")
    inv = (~mask.bool()).float()[None, None]
    pooled = F.max_pool2d(
        inv,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )
    return ~(pooled[0, 0] > 0.5)


def _binary_dilate(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return mask.bool()
    if kernel % 2 == 0:
        raise ValueError("DURR boundary kernel must be odd")
    pooled = F.max_pool2d(
        mask.bool().float()[None, None],
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )
    return pooled[0, 0] > 0.5


def _weighted_o2m_estimate(
    probs: torch.Tensor,
    quality: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return one branch-balanced O2M probability estimate and one aggregate
    reliability scalar. O2M multiplicity does not increase total branch mass.
    """
    w = quality.float().clamp_min(EPS)
    w = w / w.sum().clamp_min(EPS)
    p = (probs.float() * w[:, None, None]).sum(dim=0)
    return p, w


def _max_prob_evidence(
    rows: torch.Tensor,
    proto: torch.Tensor,
    input_h: int,
    input_w: int,
    conf_floor: float,
) -> torch.Tensor:
    """
    Conservative raw Teacher evidence for safe-background construction.
    Uses low confidence floor, not only accepted pseudo labels.
    """
    mh, mw = int(proto.shape[-2]), int(proto.shape[-1])
    if rows.numel() == 0:
        return proto.new_zeros((mh, mw))

    rows = rows[rows[:, 4] >= conf_floor]
    if rows.numel() == 0:
        return proto.new_zeros((mh, mw))

    probs = mask_probs_from_coefficients(
        rows,
        proto,
        input_h,
        input_w,
    )
    if probs.numel() == 0:
        return proto.new_zeros((mh, mw))
    return probs.amax(dim=0)


@torch.no_grad()
def build_durr_routes(
    teacher,
    weak_imgs: torch.Tensor,
    final_pseudo_labels: list[torch.Tensor],
    *,
    tau_o2o: float = 0.5,
    tau_o2m: float = 0.5,
    tau_dup: float = 0.7,
    tau_match: float = 0.5,
    max_witnesses: int = 5,
    mask_threshold: float = 0.5,
    stability_low: float = 0.40,
    stability_high: float = 0.60,
    reliability_threshold: float = 0.744898,
    min_mask_pixels: int = 16,
    boundary_kernel: int = 5,
    direction_min_abs: float = 0.03,
    rescue_conf: float = 0.80,
    rescue_stability: float = 0.80,
    evidence_conf_floor: float = 0.10,
    safe_bg_teacher_prob: float = 0.15,
) -> tuple[list[dict[str, torch.Tensor | bool]], dict[str, Any]]:
    """
    Build label-free routing maps from native Teacher O2O/O2M outputs.

    NOTE: This performs one Teacher forward. generate_durr_pseudo_masks()
    intentionally keeps the old BDL pseudo construction unchanged and calls
    this function separately. The extra Teacher forward makes DURR-v1 easy to
    audit. It can be fused into one forward later after validation.
    """
    teacher.eval()
    outputs = teacher(
        weak_imgs,
        augment=False,
        visualize=False,
    )
    if not (isinstance(outputs, tuple) and len(outputs) == 2):
        raise RuntimeError("Unexpected Teacher output in DURR")

    first, branches = outputs
    if not (
        isinstance(first, tuple)
        and len(first) == 2
        and isinstance(branches, dict)
    ):
        raise RuntimeError("Expected ((O2O, proto), branches) in DURR")

    final_o2o, proto = first
    if isinstance(proto, (tuple, list)):
        proto = proto[0]

    head = teacher.model[-1]
    decoded_o2m = head._inference(
        branches["one2many"]
    ).permute(0, 2, 1)
    final_o2m = head.postprocess(decoded_o2m)

    input_h = int(weak_imgs.shape[2])
    input_w = int(weak_imgs.shape[3])

    routes: list[dict[str, torch.Tensor | bool]] = []
    totals: dict[str, Any] = {
        "durr_direction_images": 0,
        "durr_direction_pixels": 0,
        "durr_signed_abs_sum": 0.0,
        "durr_signed_abs_count": 0,
        "durr_positive_direction_pixels": 0,
        "durr_negative_direction_pixels": 0,
        "durr_rescue_images": 0,
        "durr_rescue_instances": 0,
        "durr_rescue_pixels": 0,
        "durr_teacher_empty_images": 0,
        "durr_safe_bg_pixels": 0,
    }

    for i in range(weak_imgs.shape[0]):
        rows_o2o_all = final_o2o[i]
        rows_o2m_all = final_o2m[i]

        # Low-threshold evidence is used ONLY for safe-background gating.
        o2o_evidence = _max_prob_evidence(
            rows_o2o_all,
            proto[i],
            input_h,
            input_w,
            evidence_conf_floor,
        )
        o2m_evidence = _max_prob_evidence(
            rows_o2m_all,
            proto[i],
            input_h,
            input_w,
            evidence_conf_floor,
        )

        anchors = rows_o2o_all[rows_o2o_all[:, 4] >= tau_o2o]
        candidates = rows_o2m_all[rows_o2m_all[:, 4] >= tau_o2m]

        anchor_probs = mask_probs_from_coefficients(
            anchors,
            proto[i],
            input_h,
            input_w,
        )
        anchor_masks = anchor_probs >= mask_threshold
        if anchors.numel():
            keep = anchor_masks.sum(dim=(1, 2)) >= min_mask_pixels
            anchors = anchors[keep]
            anchor_probs = anchor_probs[keep]
            anchor_masks = anchor_masks[keep]

        anchor_stab = mask_stability(
            anchor_probs,
            low=stability_low,
            high=stability_high,
        )
        anchor_rel = torch.sqrt(
            (anchors[:, 4] * anchor_stab).clamp_min(0.0)
        )

        candidate_probs = mask_probs_from_coefficients(
            candidates,
            proto[i],
            input_h,
            input_w,
        )
        candidate_masks = candidate_probs >= mask_threshold
        if candidates.numel():
            keep = candidate_masks.sum(dim=(1, 2)) >= min_mask_pixels
            candidates = candidates[keep]
            candidate_probs = candidate_probs[keep]
            candidate_masks = candidate_masks[keep]

        candidate_stab = mask_stability(
            candidate_probs,
            low=stability_low,
            high=stability_high,
        )
        candidate_quality = (
            candidates[:, 4] * candidate_stab
        ).clamp_min(0.0)
        candidate_rel = torch.sqrt(candidate_quality)

        mh, mw = int(proto.shape[-2]), int(proto.shape[-1])
        signed_map = proto.new_zeros((mh, mw))
        direction_weight = proto.new_zeros((mh, mw))
        anchor_reference = proto.new_zeros((mh, mw))
        direction_band = torch.zeros(
            (mh, mw), device=proto.device, dtype=torch.bool
        )

        best_iou, best_anchor_idx = _same_class_best_iou(
            candidates,
            anchors,
        )
        matched = (
            (best_anchor_idx >= 0)
            & (best_iou >= tau_match)
        )

        # ------------------------------------------------------------
        # A. Signed native cross-head boundary routing.
        # ------------------------------------------------------------
        for anchor_idx in range(anchors.shape[0]):
            witness_ids = torch.where(
                matched & (best_anchor_idx == anchor_idx)
            )[0]
            if witness_ids.numel() == 0:
                continue

            selection = (
                best_iou[witness_ids]
                * candidate_rel[witness_ids]
            )
            order = torch.argsort(selection, descending=True)
            witness_ids = witness_ids[order]
            if max_witnesses > 0:
                witness_ids = witness_ids[:max_witnesses]

            wp = candidate_probs[witness_ids]
            wq = candidate_quality[witness_ids]
            o2m_est, norm_w = _weighted_o2m_estimate(wp, wq)
            witness_rel = (
                candidate_rel[witness_ids] * norm_w
            ).sum().clamp(0.0, 1.0)

            po = anchor_probs[anchor_idx].float()
            ro = anchor_rel[anchor_idx].float().clamp(0.0, 1.0)
            rm = witness_rel.float()

            signed = (
                rm * o2m_est.float()
                - ro * po
            ).clamp(-1.0, 1.0)

            hard = anchor_masks[anchor_idx]
            eroded = _binary_erode(hard, boundary_kernel)
            dilated = _binary_dilate(hard, boundary_kernel)
            band = dilated & (~eroded)

            abs_signed = signed.abs()
            mask = band & (abs_signed >= direction_min_abs)
            if not mask.any():
                continue

            reliability_pair = torch.sqrt(
                (ro * rm).clamp_min(0.0)
            )
            local_weight = (
                abs_signed * reliability_pair
            ).clamp(0.0, 1.0)

            # If multiple objects overlap, keep the route with higher weight.
            replace = mask & (local_weight > direction_weight)
            signed_map[replace] = signed[replace]
            direction_weight[replace] = local_weight[replace]
            anchor_reference[replace] = po[replace]
            direction_band |= mask

        if direction_band.any():
            vals = signed_map[direction_band]
            totals["durr_direction_images"] += 1
            totals["durr_direction_pixels"] += int(direction_band.sum().item())
            totals["durr_positive_direction_pixels"] += int((vals > 0).sum().item())
            totals["durr_negative_direction_pixels"] += int((vals < 0).sum().item())
            totals["durr_signed_abs_sum"] += float(vals.abs().sum().item())
            totals["durr_signed_abs_count"] += int(vals.numel())

        # ------------------------------------------------------------
        # B. Reliable O2M rescue (only when native O2O has no anchor).
        # This does NOT become fake O2O for BDL.
        # ------------------------------------------------------------
        rescue_prob = proto.new_zeros((mh, mw))
        rescue_mask = torch.zeros(
            (mh, mw), device=proto.device, dtype=torch.bool
        )
        rescue_count = 0

        if anchors.shape[0] == 0 and candidates.shape[0] > 0:
            rescue_ok = (
                (candidates[:, 4] >= rescue_conf)
                & (candidate_stab >= rescue_stability)
                & (candidate_rel >= reliability_threshold)
            )
            rescue_ids = torch.where(rescue_ok)[0]

            if rescue_ids.numel():
                rescue_rows = candidates[rescue_ids]
                keep = classwise_nms_indices(
                    rescue_rows,
                    tau_dup,
                )
                rescue_ids = rescue_ids[keep]

                if rescue_ids.numel():
                    rp = candidate_probs[rescue_ids]
                    rescue_prob = rp.amax(dim=0)
                    rescue_mask = rescue_prob >= mask_threshold
                    rescue_count = int(rescue_ids.numel())

        if rescue_mask.any():
            totals["durr_rescue_images"] += 1
            totals["durr_rescue_instances"] += rescue_count
            totals["durr_rescue_pixels"] += int(rescue_mask.sum().item())

        # ------------------------------------------------------------
        # C. Safe background for hallucination suppression.
        # Final Teacher pseudo empty != whole image background.
        # ------------------------------------------------------------
        teacher_empty = (
            i >= len(final_pseudo_labels)
            or final_pseudo_labels[i].shape[0] == 0
        )
        if teacher_empty:
            totals["durr_teacher_empty_images"] += 1

        safe_bg = (
            (o2o_evidence < safe_bg_teacher_prob)
            & (o2m_evidence < safe_bg_teacher_prob)
        )
        totals["durr_safe_bg_pixels"] += int(safe_bg.sum().item())

        routes.append(
            {
                "signed_direction": signed_map.detach(),
                "direction_weight": direction_weight.detach(),
                "direction_band": direction_band.detach(),
                "anchor_reference": anchor_reference.detach(),
                "rescue_prob": rescue_prob.detach(),
                "rescue_mask": rescue_mask.detach(),
                "safe_background": safe_bg.detach(),
                "teacher_o2o_evidence": o2o_evidence.detach(),
                "teacher_o2m_evidence": o2m_evidence.detach(),
                "teacher_empty": bool(teacher_empty),
            }
        )

    totals["durr_signed_abs_mean"] = (
        totals["durr_signed_abs_sum"]
        / max(totals["durr_signed_abs_count"], 1)
    )
    return routes, totals


@torch.no_grad()
def generate_durr_pseudo_masks(
    teacher,
    weak_imgs: torch.Tensor,
    *,
    tau_o2o: float = 0.5,
    tau_o2m: float = 0.5,
    tau_no: float = 0.2,
    tau_dup: float = 0.7,
    tau_match: float = 0.5,
    max_witnesses: int = 5,
    mask_threshold: float = 0.5,
    stability_low: float = 0.40,
    stability_high: float = 0.60,
    reliability_threshold: float = 0.744898,
    min_mask_pixels: int = 16,
    bdl_boundary_kernel: int = 3,
    durr_boundary_kernel: int = 5,
    durr_direction_min_abs: float = 0.03,
    durr_rescue_conf: float = 0.80,
    durr_rescue_stability: float = 0.80,
    durr_evidence_conf_floor: float = 0.10,
    durr_safe_bg_teacher_prob: float = 0.15,
):
    """
    DURR-v1 deliberately leaves the validated BDL/Mask-DHF pseudo population
    unchanged. DURR adds routed auxiliary supervision on top.

    Returns:
      labels
      supervision_masks
      geometry_masks
      instance_reliability
      totals
      routes
    """
    base = generate_boundary_dhf_pseudo_masks(
        teacher=teacher,
        weak_imgs=weak_imgs,
        tau_o2o=tau_o2o,
        tau_o2m=tau_o2m,
        tau_no=tau_no,
        tau_dup=tau_dup,
        tau_match=tau_match,
        max_witnesses=max_witnesses,
        mask_threshold=mask_threshold,
        stability_low=stability_low,
        stability_high=stability_high,
        reliability_threshold=reliability_threshold,
        min_mask_pixels=min_mask_pixels,
        boundary_kernel=bdl_boundary_kernel,
        return_debug=False,
    )
    (
        labels,
        supervision_masks,
        geometry_masks,
        instance_reliability,
        totals,
    ) = base

    routes, durr_totals = build_durr_routes(
        teacher=teacher,
        weak_imgs=weak_imgs,
        final_pseudo_labels=labels,
        tau_o2o=tau_o2o,
        tau_o2m=tau_o2m,
        tau_dup=tau_dup,
        tau_match=tau_match,
        max_witnesses=max_witnesses,
        mask_threshold=mask_threshold,
        stability_low=stability_low,
        stability_high=stability_high,
        reliability_threshold=reliability_threshold,
        min_mask_pixels=min_mask_pixels,
        boundary_kernel=durr_boundary_kernel,
        direction_min_abs=durr_direction_min_abs,
        rescue_conf=durr_rescue_conf,
        rescue_stability=durr_rescue_stability,
        evidence_conf_floor=durr_evidence_conf_floor,
        safe_bg_teacher_prob=durr_safe_bg_teacher_prob,
    )
    totals = dict(totals)
    totals.update(durr_totals)
    return (
        labels,
        supervision_masks,
        geometry_masks,
        instance_reliability,
        totals,
        routes,
    )


def extract_student_semantic_logits(
    student_outputs,
) -> torch.Tensor:
    """
    Extract differentiable Proto26 semantic foreground logits from the native
    one-to-many branch. Expected shape: [B, C, H, W]. For single-class polyp,
    C=1.
    """
    if not isinstance(student_outputs, dict):
        raise RuntimeError(
            f"DURR expected Student training output dict, got {type(student_outputs)}"
        )

    branch = student_outputs.get("one2many")
    if not isinstance(branch, dict):
        raise RuntimeError("DURR missing Student one2many branch")

    proto = branch.get("proto")
    if not (
        isinstance(proto, (tuple, list))
        and len(proto) == 2
        and isinstance(proto[1], torch.Tensor)
    ):
        raise RuntimeError(
            "DURR requires Proto26 semantic logits in one2many['proto']=(proto, sem_logits)"
        )

    logits = proto[1]
    if logits.ndim != 4:
        raise RuntimeError(
            f"DURR semantic logits must be [B,C,H,W], got {tuple(logits.shape)}"
        )
    if logits.shape[1] != 1:
        raise RuntimeError(
            "DURR-v1 is implemented for single-class polyp segmentation "
            f"(got C={logits.shape[1]})."
        )
    return logits[:, 0]


def _resize_route_map(
    x: torch.Tensor,
    hw: tuple[int, int],
    *,
    binary: bool = False,
) -> torch.Tensor:
    y = x.float()[None, None]
    if tuple(y.shape[-2:]) == tuple(hw):
        out = y[0, 0]
    elif binary:
        out = F.interpolate(
            y,
            size=hw,
            mode="nearest",
        )[0, 0]
    else:
        out = F.interpolate(
            y,
            size=hw,
            mode="bilinear",
            align_corners=False,
        )[0, 0]
    return out


def compute_durr_loss(
    student_outputs,
    routes: list[dict[str, torch.Tensor | bool]],
    *,
    direction_margin: float = 0.05,
    lambda_direction: float = 0.20,
    lambda_rescue: float = 0.50,
    lambda_hallucination: float = 0.20,
    hall_student_prob: float = 0.70,
    hall_area_threshold: float = 0.10,
    warmup_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute differentiable DURR losses on Student semantic logits.

    L_dir:
      signed boundary direction hinge.
    L_rescue:
      positive semantic retention on promoted O2M rescue masks.
    L_hall:
      safe-background suppression only when Teacher final pseudo is empty AND
      Student high-confidence foreground occupies an abnormally large area.
    """
    logits = extract_student_semantic_logits(student_outputs)
    if len(routes) != logits.shape[0]:
        raise RuntimeError(
            f"DURR routes={len(routes)} != Student batch={logits.shape[0]}"
        )

    device = logits.device
    hw = tuple(logits.shape[-2:])
    zero = logits.sum() * 0.0

    dir_num = zero
    dir_den = zero
    rescue_num = zero
    rescue_den = zero
    hall_num = zero
    hall_den = zero

    direction_pixels = 0
    rescue_pixels = 0
    hall_pixels = 0
    hall_trigger_images = 0
    student_fg_area_sum = 0.0

    for i, route in enumerate(routes):
        logit = logits[i]
        prob = logit.sigmoid()

        # --------------------------------------------------------
        # Signed Boundary Routing
        # --------------------------------------------------------
        signed = _resize_route_map(
            route["signed_direction"].to(device),
            hw,
        )
        weight = _resize_route_map(
            route["direction_weight"].to(device),
            hw,
        )
        band = _resize_route_map(
            route["direction_band"].to(device),
            hw,
            binary=True,
        ) > 0.5
        ref = _resize_route_map(
            route["anchor_reference"].to(device),
            hw,
        ).clamp(0.0, 1.0)

        active = band & (weight > 0)
        if active.any():
            sign = torch.sign(signed[active]).detach()
            progress = sign * (prob[active] - ref[active])
            w = weight[active].detach().clamp_min(EPS)
            l = F.relu(direction_margin - progress)
            dir_num = dir_num + (w * l).sum()
            dir_den = dir_den + w.sum()
            direction_pixels += int(active.sum().item())

        # --------------------------------------------------------
        # Reliable O2M Rescue Retention
        # --------------------------------------------------------
        rescue_mask = _resize_route_map(
            route["rescue_mask"].to(device),
            hw,
            binary=True,
        ) > 0.5
        if rescue_mask.any():
            rescue_target = _resize_route_map(
                route["rescue_prob"].to(device),
                hw,
            ).clamp(0.5, 1.0)
            l = F.binary_cross_entropy_with_logits(
                logit[rescue_mask],
                rescue_target[rescue_mask].detach(),
                reduction="sum",
            )
            rescue_num = rescue_num + l
            rescue_den = rescue_den + rescue_mask.sum().float()
            rescue_pixels += int(rescue_mask.sum().item())

        # --------------------------------------------------------
        # Safe Hallucination Suppression
        # --------------------------------------------------------
        if bool(route["teacher_empty"]):
            high_fg = prob.detach() >= hall_student_prob
            area = float(high_fg.float().mean().item())
            student_fg_area_sum += area

            if area > hall_area_threshold:
                safe = _resize_route_map(
                    route["safe_background"].to(device),
                    hw,
                    binary=True,
                ) > 0.5
                active_h = safe & high_fg
                if active_h.any():
                    hall_trigger_images += 1
                    target0 = torch.zeros_like(logit[active_h])
                    l = F.binary_cross_entropy_with_logits(
                        logit[active_h],
                        target0,
                        reduction="sum",
                    )
                    hall_num = hall_num + l
                    hall_den = hall_den + active_h.sum().float()
                    hall_pixels += int(active_h.sum().item())

    l_dir = dir_num / dir_den.clamp_min(1.0)
    l_rescue = rescue_num / rescue_den.clamp_min(1.0)
    l_hall = hall_num / hall_den.clamp_min(1.0)

    scale = float(max(0.0, min(1.0, warmup_scale)))
    total = scale * (
        float(lambda_direction) * l_dir
        + float(lambda_rescue) * l_rescue
        + float(lambda_hallucination) * l_hall
    )

    stats = {
        "durr_loss": float(total.detach().item()),
        "durr_dir_loss": float(l_dir.detach().item()),
        "durr_rescue_loss": float(l_rescue.detach().item()),
        "durr_hall_loss": float(l_hall.detach().item()),
        "durr_direction_pixels_student": float(direction_pixels),
        "durr_rescue_pixels_student": float(rescue_pixels),
        "durr_hall_pixels_student": float(hall_pixels),
        "durr_hall_trigger_images": float(hall_trigger_images),
        "durr_teacher_empty_student_area_sum": float(student_fg_area_sum),
        "durr_warmup_scale": scale,
    }
    return total, stats
