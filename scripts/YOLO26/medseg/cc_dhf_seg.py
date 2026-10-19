#!/usr/bin/env python3

from __future__ import annotations

import torch

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
    """Best same-class anchor IoU for every O2M candidate."""
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


def _branch_balanced_consensus(
    anchor_prob: torch.Tensor,
    anchor_quality: torch.Tensor,
    witness_probs: torch.Tensor,
    witness_quality: torch.Tensor,
) -> torch.Tensor:
    """
    Branch-balanced O2O/O2M consensus.

    The audit showed many O2M witnesses per O2O anchor (median ~9). Summing all
    witness weights directly would let one O2M branch dominate the single O2O
    vote purely because it produces more assignments. We therefore first form
    one reliability-weighted O2M branch estimate, then fuse that estimate with
    the O2O mask using one quality score per branch.
    """
    if witness_probs.shape[0] == 0:
        return anchor_prob

    w = witness_quality.clamp_min(EPS)
    w_sum = w.sum().clamp_min(EPS)

    o2m_prob = (
        witness_probs
        * w[:, None, None]
    ).sum(dim=0) / w_sum

    # One branch-level O2M quality, independent of witness multiplicity.
    o2m_quality = w.mean()
    o2o_quality = anchor_quality.clamp_min(EPS)

    return (
        o2o_quality * anchor_prob
        + o2m_quality * o2m_prob
    ) / (o2o_quality + o2m_quality).clamp_min(EPS)


@torch.no_grad()
def generate_cc_dhf_pseudo_masks(
    teacher,
    weak_imgs: torch.Tensor,
    tau_o2o: float = 0.5,
    tau_o2m: float = 0.5,
    tau_no: float = 0.2,
    tau_dup: float = 0.7,
    tau_match: float = 0.5,
    max_consensus_witnesses: int = 5,
    mask_threshold: float = 0.5,
    stability_low: float = 0.40,
    stability_high: float = 0.60,
    reliability_threshold: float = 0.744898,
    min_mask_pixels: int = 16,
    return_instance_reliability: bool = False,
):
    """
    CC-DHF v1: Consensus-and-Coverage Dual-Head Fusion.

    CONSENSUS ROLE
    --------------
    O2M predictions matched to an O2O anchor (same class, box IoU >= tau_match)
    are no longer discarded as duplicates. Up to max_consensus_witnesses are
    selected by match-IoU * reliability and used to refine only the O2O pseudo
    mask. The O2O pseudo box/class/confidence remain unchanged.

    COVERAGE ROLE
    -------------
    O2M predictions that remain genuinely novel versus all same-class O2O
    anchors (best IoU <= tau_no) follow the original Mask-DHF path:
      classwise NMS -> threshold stability -> reliability gate -> extras.

    The gray zone tau_no < IoU < tau_match is ignored, preserving the original
    conservative duplicate handling while cleanly separating consensus from
    coverage.

    No target labels or target GT masks are used.
    """
    if tau_no > tau_match:
        raise ValueError(
            "CC-DHF requires tau_no <= tau_match so coverage and consensus "
            "regions do not overlap."
        )

    if max_consensus_witnesses < 0:
        raise ValueError("max_consensus_witnesses must be >= 0")

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

    labels_out = []
    masks_out = []
    instance_reliability_out = []

    totals = {
        "anchors": 0,
        "candidates": 0,
        "consensus_anchors": 0,
        "consensus_witnesses": 0,
        "consensus_fallbacks": 0,
        "consensus_abs_shift_sum": 0.0,
        "consensus_abs_shift_count": 0,
        "consensus_mask_iou_sum": 0.0,
        "consensus_mask_iou_count": 0,
        "coverage_candidates": 0,
        "box_novel": 0,
        "box_dhf_extras": 0,
        "mask_dhf_extras": 0,
        "rejected_reliability": 0,
        "pseudo": 0,
    }

    # Diagnostic list matching the legacy Mask-DHF aggregate convention:
    # coverage-candidate reliabilities after coverage NMS, before rejection.
    reliabilities = []

    for i in range(weak_imgs.shape[0]):
        anchors = final_o2o[i]
        candidates = final_o2m[i]

        anchors = anchors[
            anchors[:, 4] >= tau_o2o
        ]
        candidates = candidates[
            candidates[:, 4] >= tau_o2m
        ]

        # --------------------------------------------------------
        # O2O anchors: valid masks + quality metadata
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # O2M candidates: confidence + valid mask only.
        # We intentionally delay novelty rejection because matched
        # candidates are now useful consensus witnesses.
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # Split O2M into CONSENSUS, COVERAGE, and gray-zone rejects.
        # --------------------------------------------------------
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
            # Different-class candidates have best_iou=-1 and are novel.
            coverage = best_iou <= tau_no

        # --------------------------------------------------------
        # CONSENSUS: refine O2O masks, preserve O2O boxes.
        # --------------------------------------------------------
        refined_anchor_masks = anchor_masks.float().clone()

        for anchor_idx in range(
            anchors.shape[0]
        ):
            witness_ids = torch.where(
                matched
                & (best_anchor_idx == anchor_idx)
            )[0]

            if witness_ids.numel() == 0:
                continue

            # Match quality * reliability favors same-instance geometry
            # and stable masks. Selection only; fusion weights below use
            # confidence*stability as defined in the proposal.
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
                max_consensus_witnesses > 0
                and witness_ids.numel()
                > max_consensus_witnesses
            ):
                witness_ids = witness_ids[
                    :max_consensus_witnesses
                ]

            consensus_prob = (
                _branch_balanced_consensus(
                    anchor_prob=anchor_probs[
                        anchor_idx
                    ],
                    anchor_quality=anchor_quality[
                        anchor_idx
                    ],
                    witness_probs=candidate_probs[
                        witness_ids
                    ],
                    witness_quality=candidate_quality[
                        witness_ids
                    ],
                )
            )

            consensus_mask = (
                consensus_prob
                >= mask_threshold
            )

            # Never drop a trusted O2O anchor because of consensus.
            if (
                int(consensus_mask.sum().item())
                < min_mask_pixels
            ):
                consensus_mask = anchor_masks[
                    anchor_idx
                ]
                totals[
                    "consensus_fallbacks"
                ] += 1
            else:
                original_mask = anchor_masks[anchor_idx]
                shift_region = (
                    original_mask | consensus_mask
                )
                abs_shift = (
                    consensus_prob
                    - anchor_probs[anchor_idx]
                ).abs()
                if shift_region.any():
                    shift = abs_shift[shift_region].mean()
                else:
                    shift = abs_shift.mean()

                inter = (
                    original_mask & consensus_mask
                ).sum().float()
                union = (
                    original_mask | consensus_mask
                ).sum().float()
                consensus_iou = (
                    inter / union.clamp_min(1.0)
                )

                totals[
                    "consensus_abs_shift_sum"
                ] += float(shift.item())
                totals[
                    "consensus_abs_shift_count"
                ] += 1
                totals[
                    "consensus_mask_iou_sum"
                ] += float(consensus_iou.item())
                totals[
                    "consensus_mask_iou_count"
                ] += 1

            refined_anchor_masks[
                anchor_idx
            ] = consensus_mask.float()

            totals[
                "consensus_anchors"
            ] += 1
            totals[
                "consensus_witnesses"
            ] += int(
                witness_ids.numel()
            )

        # --------------------------------------------------------
        # COVERAGE: original Mask-DHF on genuinely novel O2M.
        # --------------------------------------------------------
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
            extra_masks = (
                extra_probs >= mask_threshold
            )
            extra_reliability = (
                coverage_reliability[
                    reliable
                ]
            )
        else:
            extras = coverage_candidates
            extra_masks = torch.zeros(
                (
                    0,
                    proto.shape[-2],
                    proto.shape[-1],
                ),
                device=proto.device,
                dtype=torch.bool,
            )
            extra_reliability = torch.zeros(
                (0,),
                device=proto.device,
                dtype=proto.dtype,
            )

        totals["mask_dhf_extras"] += int(
            extras.shape[0]
        )

        # --------------------------------------------------------
        # Final fusion: consensus-refined anchors + coverage extras.
        # --------------------------------------------------------
        if anchors.numel() and extras.numel():
            fused = torch.cat(
                [anchors, extras],
                dim=0,
            )
            fused_masks = torch.cat(
                [
                    refined_anchor_masks,
                    extra_masks.float(),
                ],
                dim=0,
            )
            # CC-DHF v1 changes mask fusion only. Keep anchor reliability
            # unchanged so optional downstream reliability experiments do
            # not become a second confound.
            fused_reliability = torch.cat(
                [
                    anchor_reliability,
                    extra_reliability,
                ],
                dim=0,
            )
        elif anchors.numel():
            fused = anchors
            fused_masks = refined_anchor_masks
            fused_reliability = anchor_reliability
        else:
            fused = extras
            fused_masks = extra_masks.float()
            fused_reliability = extra_reliability

        if fused.numel():
            order = torch.argsort(
                fused[:, 4],
                descending=True,
            )
            fused = fused[order]
            fused_masks = fused_masks[order]
            fused_reliability = (
                fused_reliability[order]
            )

        if not (
            fused.shape[0]
            == fused_masks.shape[0]
            == fused_reliability.shape[0]
        ):
            raise RuntimeError(
                "CC-DHF final box/mask/reliability alignment failure"
            )

        totals["pseudo"] += int(
            fused.shape[0]
        )

        labels_out.append(fused)
        masks_out.append(fused_masks)
        instance_reliability_out.append(
            fused_reliability
        )

    totals["reliabilities"] = reliabilities
    totals["consensus_abs_shift_mean"] = (
        totals["consensus_abs_shift_sum"]
        / max(
            totals["consensus_abs_shift_count"],
            1,
        )
    )
    totals["consensus_mask_iou_mean"] = (
        totals["consensus_mask_iou_sum"]
        / max(
            totals["consensus_mask_iou_count"],
            1,
        )
    )

    if return_instance_reliability:
        return (
            labels_out,
            masks_out,
            instance_reliability_out,
            totals,
        )

    return (
        labels_out,
        masks_out,
        totals,
    )
