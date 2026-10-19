#!/usr/bin/env python3

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
    """Return best same-class O2O IoU and anchor index for each O2M candidate."""
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

    ious = box_iou(
        candidates[:, :4],
        anchors[:, :4],
    )

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


def _binary_erode(
    mask: torch.Tensor,
    kernel: int,
) -> torch.Tensor:
    """Binary erosion for one HxW mask."""
    if kernel <= 1:
        return mask.bool()
    if kernel % 2 == 0:
        raise ValueError("BDL boundary kernel must be odd")

    inv = (~mask.bool()).float()[None, None]
    pooled = F.max_pool2d(
        inv,
        kernel_size=kernel,
        stride=1,
        padding=kernel // 2,
    )
    return ~(pooled[0, 0] > 0.5)


def _weighted_o2m_estimate(
    witness_probs: torch.Tensor,
    witness_quality: torch.Tensor,
) -> torch.Tensor:
    """
    Collapse multiple O2M witnesses into ONE O2M-branch probability estimate.

    This avoids multiplicity bias: the O2M branch receives one aggregate vote,
    no matter whether an anchor has one witness or ten witnesses.
    """
    if witness_probs.shape[0] == 0:
        raise ValueError("Cannot build O2M estimate without witnesses")

    w = witness_quality.float().clamp_min(EPS)
    w = w / w.sum().clamp_min(EPS)

    return (
        witness_probs.float()
        * w[:, None, None]
    ).sum(dim=0)


def _weighted_cross_head_mad(
    anchor_prob: torch.Tensor,
    witness_probs: torch.Tensor,
    witness_quality: torch.Tensor,
) -> torch.Tensor:
    """
    Label-free pixel-wise cross-head disagreement.

        D(x) = sum_j \bar{w}_j |P_o2o(x) - P_o2m_j(x)|

    O2M weights sum to one, so disagreement magnitude is independent of the
    number of O2M assignments. D is bounded in [0, 1].
    """
    if witness_probs.shape[0] == 0:
        return torch.zeros_like(anchor_prob, dtype=torch.float32)

    w = witness_quality.float().clamp_min(EPS)
    w = w / w.sum().clamp_min(EPS)

    return (
        (witness_probs.float() - anchor_prob.float()).abs()
        * w[:, None, None]
    ).sum(dim=0).clamp_(0.0, 1.0)


def _branch_balanced_std(
    anchor_prob: torch.Tensor,
    witness_probs: torch.Tensor,
    witness_quality: torch.Tensor,
) -> torch.Tensor:
    """
    Diagnostic cross-head predictive standard deviation.

    Total mass is balanced 50/50 between O2O and the whole O2M branch, while
    O2M mass is distributed among witnesses according to confidence*stability.
    This is reported for analysis but is not the optimization signal in v1.
    """
    if witness_probs.shape[0] == 0:
        return torch.zeros_like(anchor_prob, dtype=torch.float32)

    w = witness_quality.float().clamp_min(EPS)
    w = w / w.sum().clamp_min(EPS)

    o2m_mean = (
        witness_probs.float()
        * w[:, None, None]
    ).sum(dim=0)

    mean = 0.5 * anchor_prob.float() + 0.5 * o2m_mean

    var_o2o = 0.5 * (anchor_prob.float() - mean).pow(2)
    var_o2m = 0.5 * (
        (witness_probs.float() - mean[None]).pow(2)
        * w[:, None, None]
    ).sum(dim=0)

    return torch.sqrt((var_o2o + var_o2m).clamp_min(0.0) + EPS)


def _soften_inner_boundary(
    hard_mask: torch.Tensor,
    anchor_prob: torch.Tensor,
    disagreement: torch.Tensor,
    boundary_kernel: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """
    Create an uncertainty-aware soft pseudo mask without changing geometry.

    Only the INNER boundary of the trusted O2O hard mask is softened:

        T(x) = (1-D(x))*M(x) + D(x)*P_o2o(x),  x in inner boundary
        T(x) = M(x),                            otherwise

    Because D in [0,1] and P_o2o>=0.5 on M, all softened boundary pixels remain
    positive. Thus the binary support used by SegMARD and semantic supervision
    is unchanged; only the instance-mask BCE target becomes softer where native
    O2O/O2M predictions disagree.
    """
    hard = hard_mask.bool()
    eroded = _binary_erode(hard, boundary_kernel)

    # Very small pseudo masks can vanish after a 3x3 erosion. Do not soften the
    # whole object in that case; preserve the original hard target.
    if hard.any() and not eroded.any():
        zero = torch.zeros_like(hard)
        return hard.float(), zero, eroded, True

    inner_boundary = hard & (~eroded)
    target = hard.float()

    if inner_boundary.any():
        alpha = disagreement.float().clamp(0.0, 1.0)
        target[inner_boundary] = (
            (1.0 - alpha[inner_boundary])
            * target[inner_boundary]
            + alpha[inner_boundary]
            * anchor_prob.float()[inner_boundary]
        )

    return target.clamp_(0.0, 1.0), inner_boundary, eroded, False


def _debug_item(
    weak_img: torch.Tensor,
    o2o_union: torch.Tensor,
    o2m_union: torch.Tensor,
    disagreement_map: torch.Tensor,
    boundary_union: torch.Tensor,
    soft_union: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "image": weak_img.detach().float().cpu(),
        "o2o_hard": o2o_union.detach().float().cpu(),
        "o2m_witness": o2m_union.detach().float().cpu(),
        "disagreement": disagreement_map.detach().float().cpu(),
        "boundary": boundary_union.detach().float().cpu(),
        "soft_target": soft_union.detach().float().cpu(),
    }


@torch.no_grad()
def generate_boundary_dhf_pseudo_masks(
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
    boundary_kernel: int = 3,
    return_debug: bool = False,
):
    """
    BDL-DHF v1: Boundary-Disagreement Learning with Mask-DHF coverage.

    Core design
    -----------
    1. O2O predictions remain trusted anchors. Their boxes/classes/confidences
       and HARD pseudo-mask geometry are unchanged from Mask-DHF.
    2. Same-class O2M predictions with box IoU >= tau_match are NOT fused into
       the O2O mask. They are used only as uncertainty witnesses.
    3. Pixel-wise cross-head disagreement is measured by a reliability-weighted
       mean absolute difference between the O2O soft mask and matched O2M soft
       masks. This signal only SOFTENS the inner O2O boundary target.
    4. Truly novel O2M predictions (best IoU <= tau_no) follow the original
       Mask-DHF coverage path: NMS -> mask stability -> reliability threshold.
    5. SegMARD should consume geometry_masks_out, never supervision_masks_out.

    No target labels or target GT masks are used.

    Returns
    -------
    labels_out:
        final pseudo rows, same population logic as Mask-DHF.
    supervision_masks_out:
        float masks in [0,1] for native instance-mask BCE. O2O inner boundaries
        may be soft; coverage extras remain hard.
    geometry_masks_out:
        binary masks for SegMARD and geometry-preserving supervision.
    instance_reliability_out:
        per-instance reliability aligned 1:1 with labels/masks.
    totals:
        aggregate diagnostics.
    debug_out (optional):
        label-free visualization tensors, one dictionary per input image.
    """
    if tau_no > tau_match:
        raise ValueError(
            "BDL-DHF requires tau_no <= tau_match so coverage and witness "
            "regions do not overlap."
        )
    if max_witnesses < 0:
        raise ValueError("max_witnesses must be >= 0")
    if boundary_kernel < 1 or boundary_kernel % 2 == 0:
        raise ValueError("boundary_kernel must be a positive odd integer")

    teacher.eval()

    outputs = teacher(
        weak_imgs,
        augment=False,
        visualize=False,
    )

    if not (
        isinstance(outputs, tuple)
        and len(outputs) == 2
    ):
        raise RuntimeError("Unexpected Teacher output")

    first, branches = outputs

    if not (
        isinstance(first, tuple)
        and len(first) == 2
    ):
        raise RuntimeError(
            "Expected ((O2O, proto), branches)"
        )

    final_o2o, proto = first

    # Proto26 may expose (instance prototypes, semantic logits).
    if isinstance(proto, (tuple, list)):
        proto = proto[0]

    if not isinstance(branches, dict):
        raise RuntimeError(
            "Missing Teacher branch dictionary"
        )

    head = teacher.model[-1]

    decoded_o2m = head._inference(
        branches["one2many"]
    ).permute(0, 2, 1)

    final_o2m = head.postprocess(
        decoded_o2m
    )

    input_h = int(weak_imgs.shape[2])
    input_w = int(weak_imgs.shape[3])

    labels_out: list[torch.Tensor] = []
    supervision_masks_out: list[torch.Tensor] = []
    geometry_masks_out: list[torch.Tensor] = []
    instance_reliability_out: list[torch.Tensor] = []
    debug_out: list[dict[str, torch.Tensor]] = []

    totals: dict[str, Any] = {
        "anchors": 0,
        "candidates": 0,
        "bdl_anchors": 0,
        "bdl_witnesses": 0,
        "bdl_boundary_fallbacks": 0,
        "bdl_boundary_pixels": 0,
        "bdl_interior_pixels": 0,
        "bdl_boundary_disagreement_sum": 0.0,
        "bdl_boundary_disagreement_count": 0,
        "bdl_interior_disagreement_sum": 0.0,
        "bdl_interior_disagreement_count": 0,
        "bdl_soft_target_shift_sum": 0.0,
        "bdl_soft_target_shift_count": 0,
        "bdl_predictive_std_sum": 0.0,
        "bdl_predictive_std_count": 0,
        "coverage_candidates": 0,
        "box_novel": 0,
        "box_dhf_extras": 0,
        "mask_dhf_extras": 0,
        "rejected_reliability": 0,
        "pseudo": 0,
    }

    # Legacy-style diagnostic: coverage reliabilities after coverage NMS and
    # before the reliability gate. Keep this separate from aligned per-instance
    # reliability output.
    reliabilities: list[float] = []

    for i in range(weak_imgs.shape[0]):
        anchors = final_o2o[i]
        candidates = final_o2m[i]

        anchors = anchors[
            anchors[:, 4] >= tau_o2o
        ]
        candidates = candidates[
            candidates[:, 4] >= tau_o2m
        ]

        # ------------------------------------------------------------
        # O2O anchors: identical trusted-anchor logic to Mask-DHF.
        # ------------------------------------------------------------
        anchor_probs = mask_probs_from_coefficients(
            anchors,
            proto[i],
            input_h,
            input_w,
        )
        anchor_masks = (
            anchor_probs >= mask_threshold
        )

        if anchors.numel():
            keep = (
                anchor_masks.sum(dim=(1, 2))
                >= min_mask_pixels
            )
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
        anchor_reliability = torch.sqrt(
            anchor_quality
        )

        totals["anchors"] += int(
            anchors.shape[0]
        )

        # ------------------------------------------------------------
        # O2M candidate masks before novelty rejection.
        # ------------------------------------------------------------
        totals["candidates"] += int(
            candidates.shape[0]
        )

        candidate_probs = mask_probs_from_coefficients(
            candidates,
            proto[i],
            input_h,
            input_w,
        )
        candidate_masks = (
            candidate_probs >= mask_threshold
        )

        if candidates.numel():
            valid = (
                candidate_masks.sum(dim=(1, 2))
                >= min_mask_pixels
            )
            candidates = candidates[valid]
            candidate_probs = candidate_probs[valid]

        candidate_stability = mask_stability(
            candidate_probs,
            low=stability_low,
            high=stability_high,
        )
        candidate_quality = (
            candidates[:, 4]
            * candidate_stability
        ).clamp_min(0.0)
        candidate_reliability = torch.sqrt(
            candidate_quality
        )

        best_iou, best_anchor_idx = (
            _same_class_best_iou(
                candidates,
                anchors,
            )
        )

        matched = (
            (best_anchor_idx >= 0)
            & (best_iou >= tau_match)
        )

        if anchors.shape[0] == 0:
            coverage = torch.ones(
                candidates.shape[0],
                device=candidates.device,
                dtype=torch.bool,
            )
        else:
            coverage = best_iou <= tau_no

        # Geometry remains the trusted O2O hard mask. Supervision starts hard
        # and is softened only on matched-anchor inner boundaries.
        geometry_anchor_masks = anchor_masks.float().clone()
        supervision_anchor_masks = anchor_masks.float().clone()

        mh = int(proto.shape[-2])
        mw = int(proto.shape[-1])

        if return_debug:
            dbg_o2o = torch.zeros((mh, mw), device=proto.device)
            dbg_o2m = torch.zeros((mh, mw), device=proto.device)
            dbg_dis = torch.zeros((mh, mw), device=proto.device)
            dbg_bnd = torch.zeros((mh, mw), device=proto.device)
            dbg_soft = torch.zeros((mh, mw), device=proto.device)

            if anchor_masks.numel():
                dbg_o2o = anchor_masks.float().amax(dim=0)

        # ------------------------------------------------------------
        # BDL: matched O2M are uncertainty witnesses, NOT pseudo-mask votes.
        # ------------------------------------------------------------
        for anchor_idx in range(
            anchors.shape[0]
        ):
            witness_ids = torch.where(
                matched
                & (best_anchor_idx == anchor_idx)
            )[0]

            if witness_ids.numel() == 0:
                continue

            selection_score = (
                best_iou[witness_ids]
                * candidate_reliability[witness_ids]
            )
            order = torch.argsort(
                selection_score,
                descending=True,
            )
            witness_ids = witness_ids[order]

            if (
                max_witnesses > 0
                and witness_ids.numel()
                > max_witnesses
            ):
                witness_ids = witness_ids[
                    :max_witnesses
                ]

            witness_probs = candidate_probs[
                witness_ids
            ]
            witness_quality = candidate_quality[
                witness_ids
            ]

            disagreement = _weighted_cross_head_mad(
                anchor_prob=anchor_probs[anchor_idx],
                witness_probs=witness_probs,
                witness_quality=witness_quality,
            )

            predictive_std = _branch_balanced_std(
                anchor_prob=anchor_probs[anchor_idx],
                witness_probs=witness_probs,
                witness_quality=witness_quality,
            )

            (
                soft_target,
                inner_boundary,
                interior,
                fallback,
            ) = _soften_inner_boundary(
                hard_mask=anchor_masks[anchor_idx],
                anchor_prob=anchor_probs[anchor_idx],
                disagreement=disagreement,
                boundary_kernel=boundary_kernel,
            )

            if fallback:
                totals["bdl_boundary_fallbacks"] += 1

            supervision_anchor_masks[
                anchor_idx
            ] = soft_target

            if inner_boundary.any():
                d_b = disagreement[
                    inner_boundary
                ]
                totals[
                    "bdl_boundary_disagreement_sum"
                ] += float(d_b.sum().item())
                totals[
                    "bdl_boundary_disagreement_count"
                ] += int(d_b.numel())
                totals[
                    "bdl_boundary_pixels"
                ] += int(d_b.numel())

                shift = (
                    soft_target
                    - geometry_anchor_masks[anchor_idx]
                ).abs()[inner_boundary]
                totals[
                    "bdl_soft_target_shift_sum"
                ] += float(shift.sum().item())
                totals[
                    "bdl_soft_target_shift_count"
                ] += int(shift.numel())

                std_b = predictive_std[
                    inner_boundary
                ]
                totals[
                    "bdl_predictive_std_sum"
                ] += float(std_b.sum().item())
                totals[
                    "bdl_predictive_std_count"
                ] += int(std_b.numel())

            if interior.any():
                d_i = disagreement[interior]
                totals[
                    "bdl_interior_disagreement_sum"
                ] += float(d_i.sum().item())
                totals[
                    "bdl_interior_disagreement_count"
                ] += int(d_i.numel())
                totals[
                    "bdl_interior_pixels"
                ] += int(d_i.numel())

            totals["bdl_anchors"] += 1
            totals["bdl_witnesses"] += int(
                witness_ids.numel()
            )

            if return_debug:
                o2m_est = _weighted_o2m_estimate(
                    witness_probs,
                    witness_quality,
                )
                dbg_o2m = torch.maximum(
                    dbg_o2m,
                    o2m_est,
                )
                pair_region = (
                    anchor_masks[anchor_idx]
                    | (o2m_est >= mask_threshold)
                )
                dbg_dis = torch.maximum(
                    dbg_dis,
                    disagreement * pair_region.float(),
                )
                dbg_bnd = torch.maximum(
                    dbg_bnd,
                    inner_boundary.float(),
                )
                dbg_soft = torch.maximum(
                    dbg_soft,
                    soft_target,
                )

        # ------------------------------------------------------------
        # COVERAGE: original Mask-DHF on genuinely novel O2M extras.
        # ------------------------------------------------------------
        coverage_candidates = candidates[
            coverage
        ]
        coverage_probs = candidate_probs[
            coverage
        ]

        totals["coverage_candidates"] += int(
            coverage_candidates.shape[0]
        )
        totals["box_novel"] += int(
            coverage_candidates.shape[0]
        )

        if coverage_candidates.numel():
            keep = classwise_nms_indices(
                coverage_candidates,
                tau_dup,
            )
            coverage_candidates = (
                coverage_candidates[keep]
            )
            coverage_probs = coverage_probs[
                keep
            ]

        totals["box_dhf_extras"] += int(
            coverage_candidates.shape[0]
        )

        if coverage_candidates.numel():
            coverage_stability = mask_stability(
                coverage_probs,
                low=stability_low,
                high=stability_high,
            )
            coverage_quality = (
                coverage_candidates[:, 4]
                * coverage_stability
            ).clamp_min(0.0)
            coverage_reliability = torch.sqrt(
                coverage_quality
            )

            reliabilities.extend(
                coverage_reliability
                .detach()
                .cpu()
                .tolist()
            )

            reliable = (
                coverage_reliability
                >= reliability_threshold
            )

            totals[
                "rejected_reliability"
            ] += int(
                (~reliable).sum().item()
            )

            extras = coverage_candidates[
                reliable
            ]
            extra_probs = coverage_probs[
                reliable
            ]
            extra_geometry_masks = (
                extra_probs >= mask_threshold
            ).float()
            extra_supervision_masks = (
                extra_geometry_masks.clone()
            )
            extra_reliability = (
                coverage_reliability[
                    reliable
                ]
            )
        else:
            extras = coverage_candidates
            extra_geometry_masks = torch.zeros(
                (0, mh, mw),
                device=proto.device,
                dtype=proto.dtype,
            )
            extra_supervision_masks = (
                extra_geometry_masks.clone()
            )
            extra_reliability = torch.zeros(
                (0,),
                device=proto.device,
                dtype=proto.dtype,
            )

        totals["mask_dhf_extras"] += int(
            extras.shape[0]
        )

        # ------------------------------------------------------------
        # Final fusion. Hard geometry and soft supervision are aligned.
        # ------------------------------------------------------------
        if anchors.numel() and extras.numel():
            fused = torch.cat(
                [anchors, extras],
                dim=0,
            )
            fused_supervision_masks = torch.cat(
                [
                    supervision_anchor_masks,
                    extra_supervision_masks,
                ],
                dim=0,
            )
            fused_geometry_masks = torch.cat(
                [
                    geometry_anchor_masks,
                    extra_geometry_masks,
                ],
                dim=0,
            )
            fused_reliability = torch.cat(
                [
                    anchor_reliability,
                    extra_reliability,
                ],
                dim=0,
            )
        elif anchors.numel():
            fused = anchors
            fused_supervision_masks = (
                supervision_anchor_masks
            )
            fused_geometry_masks = (
                geometry_anchor_masks
            )
            fused_reliability = (
                anchor_reliability
            )
        else:
            fused = extras
            fused_supervision_masks = (
                extra_supervision_masks
            )
            fused_geometry_masks = (
                extra_geometry_masks
            )
            fused_reliability = (
                extra_reliability
            )

        if fused.numel():
            order = torch.argsort(
                fused[:, 4],
                descending=True,
            )
            fused = fused[order]
            fused_supervision_masks = (
                fused_supervision_masks[order]
            )
            fused_geometry_masks = (
                fused_geometry_masks[order]
            )
            fused_reliability = (
                fused_reliability[order]
            )

        if not (
            fused.shape[0]
            == fused_supervision_masks.shape[0]
            == fused_geometry_masks.shape[0]
            == fused_reliability.shape[0]
        ):
            raise RuntimeError(
                "BDL-DHF final box/supervision/geometry/reliability alignment failure"
            )

        # Strong invariant: the soft-target support never leaves hard geometry.
        # This keeps SegMARD and semantic pseudo-mask support unchanged.
        outside_mass = (
            fused_supervision_masks
            * (fused_geometry_masks <= 0.5).float()
        ).abs().max() if fused_supervision_masks.numel() else proto.new_zeros(())
        if float(outside_mass.item()) > 1e-6:
            raise RuntimeError(
                "BDL soft target leaked outside hard pseudo-mask geometry"
            )

        totals["pseudo"] += int(
            fused.shape[0]
        )

        labels_out.append(fused)
        supervision_masks_out.append(
            fused_supervision_masks
        )
        geometry_masks_out.append(
            fused_geometry_masks
        )
        instance_reliability_out.append(
            fused_reliability
        )

        if return_debug:
            if extra_geometry_masks.numel():
                dbg_soft = torch.maximum(
                    dbg_soft,
                    extra_geometry_masks.amax(dim=0),
                )
            if anchors.numel() and not dbg_soft.any():
                dbg_soft = supervision_anchor_masks.amax(dim=0)

            debug_out.append(
                _debug_item(
                    weak_img=weak_imgs[i],
                    o2o_union=dbg_o2o,
                    o2m_union=dbg_o2m,
                    disagreement_map=dbg_dis,
                    boundary_union=dbg_bnd,
                    soft_union=dbg_soft,
                )
            )

    totals["reliabilities"] = reliabilities

    totals["bdl_boundary_disagreement_mean"] = (
        totals["bdl_boundary_disagreement_sum"]
        / max(
            totals["bdl_boundary_disagreement_count"],
            1,
        )
    )
    totals["bdl_interior_disagreement_mean"] = (
        totals["bdl_interior_disagreement_sum"]
        / max(
            totals["bdl_interior_disagreement_count"],
            1,
        )
    )
    totals["bdl_boundary_interior_ratio"] = (
        totals["bdl_boundary_disagreement_mean"]
        / max(
            totals["bdl_interior_disagreement_mean"],
            EPS,
        )
    )
    totals["bdl_soft_target_shift_mean"] = (
        totals["bdl_soft_target_shift_sum"]
        / max(
            totals["bdl_soft_target_shift_count"],
            1,
        )
    )
    totals["bdl_predictive_std_mean"] = (
        totals["bdl_predictive_std_sum"]
        / max(
            totals["bdl_predictive_std_count"],
            1,
        )
    )

    result = (
        labels_out,
        supervision_masks_out,
        geometry_masks_out,
        instance_reliability_out,
        totals,
    )

    if return_debug:
        return (*result, debug_out)

    return result