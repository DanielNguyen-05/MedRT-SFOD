"""
DURR: Dual-head Uncertainty Reliability Routing for source-free segmentation.

DURR is designed to sit between Mask-DHF pseudo-label construction and the
student/SegMARD losses.

Core routing signals (all label-free):
  1) Reliable O2M Rescue:
       O2O missing + high-confidence/stable O2M extra -> explicit positive
       rescue supervision in addition to ordinary Mask-DHF inclusion.
  2) Signed Boundary Routing:
       matched native O2O/O2M masks produce a SIGNED disagreement
           delta(x) = P_o2m(x) - P_o2o(x)
       on a two-sided boundary band. Positive delta asks the student to expand;
       negative delta asks it to shrink.
  3) Safe Hallucination Suppression:
       when the final Teacher pseudo set is empty, pixels with very low evidence
       from BOTH native heads are marked safe-background. A student that creates
       an abnormally large high-confidence foreground receives a negative loss
       only on that safe-background subset, never on the whole image.

No target GT is read here.
No new inference-time parameters are introduced.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ultralytics.utils.metrics import box_iou

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
    same_class = (
        candidates[:, 5].long()[:, None]
        == anchors[:, 5].long()[None, :]
    )
    ious = torch.where(
        same_class,
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
        mask.float()[None, None],
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )
    return pooled[0, 0] > 0.5


def _mask_iou_matrix(masks: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for NxHxW binary masks."""
    n = int(masks.shape[0])
    if n == 0:
        return masks.new_zeros((0, 0), dtype=torch.float32)
    x = masks.bool().flatten(1)
    inter = (x[:, None] & x[None, :]).sum(-1).float()
    union = (x[:, None] | x[None, :]).sum(-1).float()
    return inter / union.clamp_min(1.0)


def _weighted_o2m_estimate(
    witness_probs: torch.Tensor,
    witness_quality: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return aggregate O2M probability and normalized witness weights."""
    w = witness_quality.float().clamp_min(EPS)
    w = w / w.sum().clamp_min(EPS)
    p = (witness_probs.float() * w[:, None, None]).sum(dim=0)
    return p, w


def _union_or_zero(
    masks: torch.Tensor,
    h: int,
    w: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
        return torch.zeros((h, w), dtype=dtype, device=device)
    return masks.float().amax(dim=0).to(dtype=dtype)


def extract_student_semseg_logits(
    student_outputs: dict[str, Any],
) -> torch.Tensor:
    """
    Return differentiable semantic logits Bx1xHxW from Segment26 training output.

    Segment26 stores Proto26 output in each branch. The O2M branch keeps the
    non-detached tuple (instance prototypes, semantic logits), while the O2O
    branch receives a detached copy. We therefore use O2M semantic logits as
    the shared differentiable pixel field for DURR auxiliary supervision.
    """
    if not isinstance(student_outputs, dict):
        raise RuntimeError("DURR expects training-mode dual-head output dict")
    if "one2many" not in student_outputs:
        raise RuntimeError("DURR requires native one2many branch")

    proto = student_outputs["one2many"].get("proto", None)
    if not (isinstance(proto, (tuple, list)) and len(proto) == 2):
        raise RuntimeError(
            "DURR requires Segment26 Proto26 output "
            "(instance prototypes, semantic logits)"
        )
    sem = proto[1]
    if not isinstance(sem, torch.Tensor) or sem.ndim != 4:
        raise RuntimeError("Unexpected DURR semantic-logit tensor")
    if sem.shape[1] != 1:
        raise RuntimeError(
            "DURR-v1 is implemented for the current single-class polyp setup"
        )
    return sem


@torch.no_grad()
def generate_durr_pseudo_masks(
    teacher,
    weak_imgs: torch.Tensor,
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
    boundary_kernel: int = 5,
    route_gain: float = 1.0,
    route_min_disagreement: float = 0.02,
    rescue_conf: float = 0.80,
    rescue_stability: float = 0.80,
    rescue_consensus_iou: float = 0.70,
    rescue_min_support: int = 0,
    evidence_conf: float = 0.10,
    safe_bg_teacher_prob: float = 0.10,
):
    """
    Build standard hard Mask-DHF pseudo masks plus DURR routing maps.

    Returns
    -------
    labels_out
        Final hard pseudo rows (O2O anchors + reliable novel O2M extras).
    masks_out
        Hard geometry masks used by the native student criterion and SegMARD.
    instance_reliability_out
        sqrt(confidence * threshold-stability), aligned to labels/masks.
    totals
        Aggregate label-free diagnostics.
    routes_out
        One dict per image:
          directional_target : desired probability in signed boundary band
          directional_weight : reliability * |signed disagreement|
          signed_delta       : P_o2m - P_o2o
          rescue_mask        : reliable high-trust O2M rescue foreground
          rescue_weight      : rescue confidence map
          safe_bg_mask       : consensus-low-evidence background
          teacher_evidence   : max raw-head soft evidence
          teacher_empty      : whether final pseudo set is empty
    """
    if tau_no > tau_match:
        raise ValueError("DURR requires tau_no <= tau_match")
    if boundary_kernel < 1 or boundary_kernel % 2 == 0:
        raise ValueError("boundary_kernel must be positive odd")
    if max_witnesses < 0:
        raise ValueError("max_witnesses must be >= 0")

    teacher.eval()
    outputs = teacher(
        weak_imgs,
        augment=False,
        visualize=False,
    )
    if not (isinstance(outputs, tuple) and len(outputs) == 2):
        raise RuntimeError("Unexpected Teacher output")

    first, branches = outputs
    if not (isinstance(first, tuple) and len(first) == 2):
        raise RuntimeError("Expected ((O2O, proto), branches)")

    final_o2o, proto = first
    if isinstance(proto, (tuple, list)):
        proto = proto[0]
    if not isinstance(branches, dict):
        raise RuntimeError("Missing Teacher branch dictionary")

    head = teacher.model[-1]
    decoded_o2m = head._inference(
        branches["one2many"]
    ).permute(0, 2, 1)
    final_o2m = head.postprocess(decoded_o2m)

    input_h = int(weak_imgs.shape[2])
    input_w = int(weak_imgs.shape[3])
    mh, mw = int(proto.shape[-2]), int(proto.shape[-1])

    labels_out: list[torch.Tensor] = []
    masks_out: list[torch.Tensor] = []
    instance_reliability_out: list[torch.Tensor] = []
    routes_out: list[dict[str, Any]] = []

    totals: dict[str, Any] = {
        "anchors": 0,
        "candidates": 0,
        "box_dhf_extras": 0,
        "mask_dhf_extras": 0,
        "rejected_reliability": 0,
        "pseudo": 0,
        "durr_matched_anchors": 0,
        "durr_witnesses": 0,
        "durr_direction_pixels": 0,
        "durr_abs_delta_sum": 0.0,
        "durr_abs_delta_count": 0,
        "durr_expand_pixels": 0,
        "durr_shrink_pixels": 0,
        "durr_rescue_instances": 0,
        "durr_rescue_images": 0,
        "durr_teacher_empty_images": 0,
        "durr_safe_bg_pixels": 0,
    }

    for bi in range(weak_imgs.shape[0]):
        raw_o2o = final_o2o[bi]
        raw_o2m = final_o2m[bi]

        # ------------------------------------------------------------
        # Low-threshold raw-head evidence for SAFE background.
        # This is deliberately broader than the pseudo-label thresholds.
        # ------------------------------------------------------------
        ev_o2o = raw_o2o[raw_o2o[:, 4] >= evidence_conf]
        ev_o2m = raw_o2m[raw_o2m[:, 4] >= evidence_conf]

        ev_o2o_probs = mask_probs_from_coefficients(
            ev_o2o, proto[bi], input_h, input_w
        )
        ev_o2m_probs = mask_probs_from_coefficients(
            ev_o2m, proto[bi], input_h, input_w
        )

        teacher_evidence = torch.zeros(
            (mh, mw), device=proto.device, dtype=torch.float32
        )
        if ev_o2o_probs.numel():
            teacher_evidence = torch.maximum(
                teacher_evidence,
                ev_o2o_probs.float().amax(dim=0),
            )
        if ev_o2m_probs.numel():
            teacher_evidence = torch.maximum(
                teacher_evidence,
                ev_o2m_probs.float().amax(dim=0),
            )

        safe_bg_mask = teacher_evidence < safe_bg_teacher_prob

        # ------------------------------------------------------------
        # Standard pseudo candidates.
        # ------------------------------------------------------------
        anchors = raw_o2o[raw_o2o[:, 4] >= tau_o2o]
        candidates = raw_o2m[raw_o2m[:, 4] >= tau_o2m]

        anchor_probs = mask_probs_from_coefficients(
            anchors, proto[bi], input_h, input_w
        )
        anchor_masks = anchor_probs >= mask_threshold

        if anchors.numel():
            keep = anchor_masks.sum(dim=(1, 2)) >= min_mask_pixels
            anchors = anchors[keep]
            anchor_probs = anchor_probs[keep]
            anchor_masks = anchor_masks[keep]

        anchor_stability = mask_stability(
            anchor_probs,
            low=stability_low,
            high=stability_high,
        )
        anchor_quality = (
            anchors[:, 4] * anchor_stability
        ).clamp_min(0.0)
        anchor_reliability = torch.sqrt(anchor_quality)
        totals["anchors"] += int(anchors.shape[0])

        candidate_probs = mask_probs_from_coefficients(
            candidates, proto[bi], input_h, input_w
        )
        candidate_masks = candidate_probs >= mask_threshold

        if candidates.numel():
            valid = candidate_masks.sum(dim=(1, 2)) >= min_mask_pixels
            candidates = candidates[valid]
            candidate_probs = candidate_probs[valid]
            candidate_masks = candidate_masks[valid]

        candidate_stability = mask_stability(
            candidate_probs,
            low=stability_low,
            high=stability_high,
        )
        candidate_quality = (
            candidates[:, 4] * candidate_stability
        ).clamp_min(0.0)
        candidate_reliability = torch.sqrt(candidate_quality)
        totals["candidates"] += int(candidates.shape[0])

        best_iou, best_anchor_idx = _same_class_best_iou(
            candidates, anchors
        )
        matched = (
            (best_anchor_idx >= 0)
            & (best_iou >= tau_match)
        )
        if anchors.shape[0] == 0:
            coverage = torch.ones(
                candidates.shape[0],
                dtype=torch.bool,
                device=candidates.device,
            )
        else:
            coverage = best_iou <= tau_no

        # ------------------------------------------------------------
        # Signed directional routing from native cross-head evidence.
        # ------------------------------------------------------------
        signed_delta_map = torch.zeros(
            (mh, mw), device=proto.device, dtype=torch.float32
        )
        directional_weight = torch.zeros_like(signed_delta_map)
        directional_target = torch.zeros_like(signed_delta_map)
        o2m_witness_map = torch.zeros_like(signed_delta_map)

        for anchor_idx in range(anchors.shape[0]):
            witness_ids = torch.where(
                matched & (best_anchor_idx == anchor_idx)
            )[0]
            if witness_ids.numel() == 0:
                continue

            selection_score = (
                best_iou[witness_ids]
                * candidate_reliability[witness_ids]
            )
            order = torch.argsort(selection_score, descending=True)
            witness_ids = witness_ids[order]
            if max_witnesses > 0:
                witness_ids = witness_ids[:max_witnesses]

            witness_probs = candidate_probs[witness_ids]
            witness_quality = candidate_quality[witness_ids]
            o2m_est, witness_w = _weighted_o2m_estimate(
                witness_probs,
                witness_quality,
            )
            o2m_witness_map = torch.maximum(
                o2m_witness_map,
                o2m_est.float(),
            )

            # Aggregate O2M reliability under the same normalized witness weights.
            witness_rel = candidate_reliability[witness_ids]
            o2m_rel = (witness_w * witness_rel.float()).sum()
            pair_rel = torch.sqrt(
                (
                    anchor_reliability[anchor_idx].float()
                    * o2m_rel.float()
                ).clamp_min(0.0)
            )

            delta = (
                o2m_est.float()
                - anchor_probs[anchor_idx].float()
            )

            hard = anchor_masks[anchor_idx]
            inner = _binary_erode(hard, boundary_kernel)
            outer = _binary_dilate(hard, boundary_kernel)
            band = outer & (~inner)

            active = (
                band
                & (delta.abs() >= route_min_disagreement)
            )
            if not active.any():
                continue

            local_w = delta.abs() * pair_rel
            # If several objects overlap, keep the strongest route per pixel.
            replace = active & (local_w > directional_weight)
            signed_delta_map[replace] = delta[replace]
            directional_weight[replace] = local_w[replace]
            directional_target[replace] = torch.clamp(
                anchor_probs[anchor_idx][replace].float()
                + route_gain * delta[replace],
                0.0,
                1.0,
            )

            totals["durr_matched_anchors"] += 1
            totals["durr_witnesses"] += int(witness_ids.numel())
            totals["durr_direction_pixels"] += int(active.sum().item())
            abs_d = delta[active].abs()
            totals["durr_abs_delta_sum"] += float(abs_d.sum().item())
            totals["durr_abs_delta_count"] += int(abs_d.numel())
            totals["durr_expand_pixels"] += int(
                (active & (delta > 0)).sum().item()
            )
            totals["durr_shrink_pixels"] += int(
                (active & (delta < 0)).sum().item()
            )

        # ------------------------------------------------------------
        # Original Mask-DHF coverage.
        # ------------------------------------------------------------
        coverage_candidates = candidates[coverage]
        coverage_probs = candidate_probs[coverage]
        coverage_stab = candidate_stability[coverage]

        if coverage_candidates.numel():
            keep = classwise_nms_indices(
                coverage_candidates,
                tau_dup,
            )
            coverage_candidates = coverage_candidates[keep]
            coverage_probs = coverage_probs[keep]
            coverage_stab = coverage_stab[keep]

        totals["box_dhf_extras"] += int(
            coverage_candidates.shape[0]
        )

        if coverage_candidates.numel():
            coverage_quality = (
                coverage_candidates[:, 4]
                * coverage_stab
            ).clamp_min(0.0)
            coverage_reliability = torch.sqrt(coverage_quality)
            reliable = coverage_reliability >= reliability_threshold

            totals["rejected_reliability"] += int(
                (~reliable).sum().item()
            )

            extras = coverage_candidates[reliable]
            extra_probs = coverage_probs[reliable]
            extra_stab = coverage_stab[reliable]
            extra_rel = coverage_reliability[reliable]
            extra_masks = (
                extra_probs >= mask_threshold
            ).float()
        else:
            extras = coverage_candidates
            extra_probs = proto.new_zeros((0, mh, mw))
            extra_stab = proto.new_zeros((0,))
            extra_rel = proto.new_zeros((0,))
            extra_masks = proto.new_zeros((0, mh, mw))

        totals["mask_dhf_extras"] += int(extras.shape[0])

        # ------------------------------------------------------------
        # Reliable O2M Rescue: promotion changes AUXILIARY supervision
        # strength, not branch identity. These remain native O2M predictions.
        # ------------------------------------------------------------
        rescue_mask = torch.zeros(
            (mh, mw), device=proto.device, dtype=torch.float32
        )
        rescue_weight = torch.zeros_like(rescue_mask)
        rescue_count = 0

        if anchors.shape[0] == 0 and extras.shape[0] > 0:
            extra_consensus_support = torch.zeros(
                (extras.shape[0],),
                dtype=torch.long,
                device=extras.device,
            )
            if extras.shape[0] > 1:
                miou = _mask_iou_matrix(extra_masks > 0.5)
                eye = torch.eye(
                    extras.shape[0],
                    device=extras.device,
                    dtype=torch.bool,
                )
                support = (miou >= rescue_consensus_iou) & (~eye)
                extra_consensus_support = support.sum(dim=1)

            promote = (
                (extras[:, 4] >= rescue_conf)
                & (extra_stab >= rescue_stability)
                & (extra_consensus_support >= rescue_min_support)
            )
            if promote.any():
                promoted_masks = extra_masks[promote]
                promoted_rel = extra_rel[promote]
                # Per-pixel strongest promoted reliability.
                for mi in range(promoted_masks.shape[0]):
                    m = promoted_masks[mi] > 0.5
                    rescue_mask[m] = 1.0
                    rescue_weight[m] = torch.maximum(
                        rescue_weight[m],
                        promoted_rel[mi].float(),
                    )
                rescue_count = int(promote.sum().item())
                totals["durr_rescue_instances"] += rescue_count
                totals["durr_rescue_images"] += 1

        # ------------------------------------------------------------
        # Standard final hard pseudo population.
        # ------------------------------------------------------------
        if anchors.numel() and extras.numel():
            fused = torch.cat([anchors, extras], dim=0)
            fused_masks = torch.cat(
                [anchor_masks.float(), extra_masks],
                dim=0,
            )
            fused_rel = torch.cat(
                [anchor_reliability, extra_rel],
                dim=0,
            )
        elif anchors.numel():
            fused = anchors
            fused_masks = anchor_masks.float()
            fused_rel = anchor_reliability
        else:
            fused = extras
            fused_masks = extra_masks
            fused_rel = extra_rel

        if fused.numel():
            order = torch.argsort(fused[:, 4], descending=True)
            fused = fused[order]
            fused_masks = fused_masks[order]
            fused_rel = fused_rel[order]

        if not (
            fused.shape[0]
            == fused_masks.shape[0]
            == fused_rel.shape[0]
        ):
            raise RuntimeError("DURR pseudo alignment failure")

        teacher_empty = fused.shape[0] == 0
        if teacher_empty:
            totals["durr_teacher_empty_images"] += 1
        totals["durr_safe_bg_pixels"] += int(
            safe_bg_mask.sum().item()
        )
        totals["pseudo"] += int(fused.shape[0])

        labels_out.append(fused)
        masks_out.append(fused_masks)
        instance_reliability_out.append(fused_rel)

        o2o_union = _union_or_zero(
            anchor_masks.float(),
            mh,
            mw,
            dtype=torch.float32,
            device=proto.device,
        )
        fused_union = _union_or_zero(
            fused_masks,
            mh,
            mw,
            dtype=torch.float32,
            device=proto.device,
        )

        routes_out.append(
            {
                "o2o_union":
                    o2o_union.detach(),
                "o2m_witness":
                    o2m_witness_map.detach(),
                "fused_union":
                    fused_union.detach(),
                "directional_target":
                    directional_target.detach(),
                "directional_weight":
                    directional_weight.detach(),
                "signed_delta":
                    signed_delta_map.detach(),
                "rescue_mask":
                    rescue_mask.detach(),
                "rescue_weight":
                    rescue_weight.detach(),
                "safe_bg_mask":
                    safe_bg_mask.detach(),
                "teacher_evidence":
                    teacher_evidence.detach(),
                "teacher_empty":
                    bool(teacher_empty),
                "rescue_instances":
                    int(rescue_count),
            }
        )

    totals["durr_abs_delta_mean"] = (
        totals["durr_abs_delta_sum"]
        / max(totals["durr_abs_delta_count"], 1)
    )

    return (
        labels_out,
        masks_out,
        instance_reliability_out,
        totals,
        routes_out,
    )


def _resize_map(
    x: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str,
) -> torch.Tensor:
    y = x.float()[None, None]
    if tuple(y.shape[-2:]) == tuple(size):
        return y[0, 0]
    if mode == "nearest":
        return F.interpolate(
            y,
            size=size,
            mode="nearest",
        )[0, 0]
    return F.interpolate(
        y,
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def compute_directional_routing_loss(
    student_outputs: dict[str, Any],
    routes: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Reliability-weighted signed boundary target loss.

    Positive signed delta raises the target probability (expand).
    Negative signed delta lowers it (shrink).
    """
    logits = extract_student_semseg_logits(student_outputs)
    total = logits.new_zeros(())
    denom = logits.new_zeros(())
    pixels = 0
    expand = 0
    shrink = 0

    for bi, route in enumerate(routes):
        size = tuple(logits.shape[-2:])
        weight = _resize_map(
            route["directional_weight"].to(logits.device),
            size,
            mode="bilinear",
        ).clamp_min(0.0)
        target = _resize_map(
            route["directional_target"].to(logits.device),
            size,
            mode="bilinear",
        ).clamp(0.0, 1.0)
        signed = _resize_map(
            route["signed_delta"].to(logits.device),
            size,
            mode="bilinear",
        )

        active = weight > 0
        if not active.any():
            continue

        loss_map = F.binary_cross_entropy_with_logits(
            logits[bi, 0],
            target,
            reduction="none",
        )
        total = total + (loss_map * weight).sum()
        denom = denom + weight.sum()

        pixels += int(active.sum().item())
        expand += int((active & (signed > 0)).sum().item())
        shrink += int((active & (signed < 0)).sum().item())

    loss = total / denom.clamp_min(EPS)
    return loss, {
        "dir_pixels": float(pixels),
        "dir_expand_pixels": float(expand),
        "dir_shrink_pixels": float(shrink),
        "dir_weight_sum": float(denom.detach().item()),
    }


def compute_rescue_loss(
    student_outputs: dict[str, Any],
    routes: list[dict[str, Any]],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Extra positive persistence pressure for high-trust O2M rescue regions."""
    logits = extract_student_semseg_logits(student_outputs)
    total = logits.new_zeros(())
    denom = logits.new_zeros(())
    pixels = 0
    images = 0

    for bi, route in enumerate(routes):
        size = tuple(logits.shape[-2:])
        mask = _resize_map(
            route["rescue_mask"].to(logits.device),
            size,
            mode="nearest",
        ) > 0.5
        weight = _resize_map(
            route["rescue_weight"].to(logits.device),
            size,
            mode="nearest",
        ).clamp_min(0.0)

        if not mask.any():
            continue

        target = torch.ones_like(logits[bi, 0])
        loss_map = F.binary_cross_entropy_with_logits(
            logits[bi, 0],
            target,
            reduction="none",
        )
        w = weight * mask.float()
        total = total + (loss_map * w).sum()
        denom = denom + w.sum()
        pixels += int(mask.sum().item())
        images += 1

    loss = total / denom.clamp_min(EPS)
    return loss, {
        "rescue_pixels": float(pixels),
        "rescue_images": float(images),
        "rescue_weight_sum": float(denom.detach().item()),
    }


def compute_safe_hallucination_loss(
    student_outputs: dict[str, Any],
    routes: list[dict[str, Any]],
    *,
    student_threshold: float = 0.80,
    area_threshold: float = 0.10,
    area_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Suppress unsupported large foreground ONLY for Teacher-empty images.

    Important: Teacher empty is NOT treated as all-background GT.
    Pixel BCE is applied only where both heads have low evidence (safe_bg_mask)
    and the Student itself is highly foreground-confident. A soft excess-area
    term prevents giant hallucinations without forcing the whole image to zero.
    """
    logits = extract_student_semseg_logits(student_outputs)
    prob = logits.sigmoid()

    total_pixel = logits.new_zeros(())
    denom = logits.new_zeros(())
    total_area = logits.new_zeros(())
    triggered = 0
    pixels = 0

    for bi, route in enumerate(routes):
        if not bool(route["teacher_empty"]):
            continue

        p = prob[bi, 0]
        area_ratio_detached = float(
            (p.detach() >= student_threshold).float().mean().item()
        )
        if area_ratio_detached <= area_threshold:
            continue

        triggered += 1
        size = tuple(p.shape[-2:])
        safe_bg = _resize_map(
            route["safe_bg_mask"].to(logits.device),
            size,
            mode="nearest",
        ) > 0.5

        unsupported = (
            safe_bg
            & (p.detach() >= student_threshold)
        )
        if unsupported.any():
            target0 = torch.zeros_like(p)
            loss_map = F.binary_cross_entropy_with_logits(
                logits[bi, 0],
                target0,
                reduction="none",
            )
            total_pixel = total_pixel + loss_map[unsupported].sum()
            denom = denom + unsupported.float().sum()
            pixels += int(unsupported.sum().item())

        # Differentiable excess soft foreground mass. This is deliberately
        # weaker than whole-image BCE-to-zero.
        soft_area = p.mean()
        total_area = total_area + torch.relu(
            soft_area - area_threshold
        ).pow(2)

    pixel_loss = total_pixel / denom.clamp_min(EPS)
    area_loss = (
        total_area / max(triggered, 1)
        if triggered > 0
        else logits.new_zeros(())
    )
    loss = pixel_loss + area_weight * area_loss

    return loss, {
        "hall_triggered_images": float(triggered),
        "hall_pixels": float(pixels),
        "hall_pixel_loss": float(pixel_loss.detach().item()),
        "hall_area_loss": float(area_loss.detach().item()),
    }


def durr_ramp(
    global_step: int,
    steps_per_epoch: int,
    warmup_epochs: float,
) -> float:
    warmup_steps = max(
        1,
        int(warmup_epochs * steps_per_epoch),
    )
    return min(
        1.0,
        float(global_step) / float(warmup_steps),
    )
