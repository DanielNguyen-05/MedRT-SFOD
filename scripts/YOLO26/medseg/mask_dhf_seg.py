#!/usr/bin/env python3

from __future__ import annotations

import torch
import torchvision

from ultralytics.utils.metrics import box_iou


@torch.no_grad()
def mask_probs_from_coefficients(
    rows: torch.Tensor,
    proto: torch.Tensor,
    input_h: int,
    input_w: int,
):
    if rows.numel() == 0:
        mh = int(proto.shape[-2])
        mw = int(proto.shape[-1])

        return proto.new_zeros(
            (0, mh, mw)
        )

    coeff = rows[:, 6:]

    logits = torch.einsum(
        "nc,chw->nhw",
        coeff,
        proto,
    )

    probs = logits.sigmoid()

    _, mh, mw = probs.shape

    boxes = rows[:, :4].clone()

    boxes[:, [0, 2]] *= (
        float(mw) / float(input_w)
    )

    boxes[:, [1, 3]] *= (
        float(mh) / float(input_h)
    )

    ys = torch.arange(
        mh,
        device=proto.device,
        dtype=boxes.dtype,
    ).view(1, mh, 1)

    xs = torch.arange(
        mw,
        device=proto.device,
        dtype=boxes.dtype,
    ).view(1, 1, mw)

    x1 = boxes[:, 0].view(-1, 1, 1)
    y1 = boxes[:, 1].view(-1, 1, 1)
    x2 = boxes[:, 2].view(-1, 1, 1)
    y2 = boxes[:, 3].view(-1, 1, 1)

    crop = (
        (xs >= x1)
        & (xs < x2)
        & (ys >= y1)
        & (ys < y2)
    )

    return probs * crop.float()


def mask_stability(
    probs: torch.Tensor,
    low: float = 0.40,
    high: float = 0.60,
):
    if probs.shape[0] == 0:
        return probs.new_zeros((0,))

    lo = probs >= low
    hi = probs >= high

    inter = (
        lo & hi
    ).flatten(1).sum(1).float()

    union = (
        lo | hi
    ).flatten(1).sum(1).float()

    return (
        inter
        / union.clamp_min(1.0)
    )


def classwise_nms_indices(
    rows: torch.Tensor,
    iou_threshold: float,
):
    if rows.numel() == 0:
        return torch.zeros(
            0,
            device=rows.device,
            dtype=torch.long,
        )

    kept = []

    for cls in rows[:, 5].long().unique(
        sorted=True
    ):
        idx = torch.where(
            rows[:, 5].long() == cls
        )[0]

        local = torchvision.ops.nms(
            rows[idx, :4],
            rows[idx, 4],
            iou_threshold,
        )

        kept.append(
            idx[local]
        )

    if not kept:
        return torch.zeros(
            0,
            device=rows.device,
            dtype=torch.long,
        )

    return torch.cat(
        kept,
        dim=0,
    )


@torch.no_grad()
def generate_mask_dhf_pseudo_masks(
    teacher,
    weak_imgs: torch.Tensor,
    tau_o2o: float = 0.5,
    tau_o2m: float = 0.5,
    tau_no: float = 0.2,
    tau_dup: float = 0.7,
    mask_threshold: float = 0.5,
    stability_low: float = 0.40,
    stability_high: float = 0.60,
    reliability_threshold: float = 0.744898,
    min_mask_pixels: int = 16,
):
    """
    Official MedRT-SFSeg Mask-DHF v1.

    O2O:
        trusted anchors.

    O2M:
        confidence
        -> valid mask
        -> box novelty
        -> NMS
        -> mask-stability reliability.

    No target GT is used.
    """

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
        raise RuntimeError(
            "Unexpected Teacher output"
        )

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
    ).permute(
        0,
        2,
        1,
    )

    final_o2m = head.postprocess(
        decoded_o2m
    )

    input_h = int(
        weak_imgs.shape[2]
    )

    input_w = int(
        weak_imgs.shape[3]
    )

    labels_out = []
    masks_out = []

    totals = {
        "anchors": 0,
        "candidates": 0,
        "box_novel": 0,
        "box_dhf_extras": 0,
        "mask_dhf_extras": 0,
        "rejected_reliability": 0,
        "pseudo": 0,
    }

    reliabilities = []

    for i in range(
        weak_imgs.shape[0]
    ):
        anchors = final_o2o[i]

        candidates = final_o2m[i]

        anchors = anchors[
            anchors[:, 4]
            >= tau_o2o
        ]

        candidates = candidates[
            candidates[:, 4]
            >= tau_o2m
        ]

        # --------------------------------------------------------
        # O2O pseudo masks
        # --------------------------------------------------------

        anchor_probs = (
            mask_probs_from_coefficients(
                anchors,
                proto[i],
                input_h,
                input_w,
            )
        )

        anchor_masks = (
            anchor_probs
            >= mask_threshold
        )

        if anchors.numel():
            keep = (
                anchor_masks.sum(
                    dim=(1, 2)
                )
                >= min_mask_pixels
            )

            anchors = anchors[keep]
            anchor_masks = (
                anchor_masks[keep]
            )

        totals["anchors"] += int(
            anchors.shape[0]
        )

        totals["candidates"] += int(
            candidates.shape[0]
        )

        # --------------------------------------------------------
        # Candidate masks
        # --------------------------------------------------------

        candidate_probs = (
            mask_probs_from_coefficients(
                candidates,
                proto[i],
                input_h,
                input_w,
            )
        )

        candidate_masks = (
            candidate_probs
            >= mask_threshold
        )

        if candidates.numel():
            valid = (
                candidate_masks.sum(
                    dim=(1, 2)
                )
                >= min_mask_pixels
            )

            candidates = candidates[
                valid
            ]

            candidate_probs = (
                candidate_probs[valid]
            )

        # --------------------------------------------------------
        # Box-DHF novelty
        # --------------------------------------------------------

        if (
            candidates.numel()
            and anchors.numel()
        ):
            max_iou = box_iou(
                candidates[:, :4],
                anchors[:, :4],
            ).max(
                dim=1
            ).values

            novel = (
                max_iou <= tau_no
            )

            candidates = candidates[
                novel
            ]

            candidate_probs = (
                candidate_probs[novel]
            )

        totals["box_novel"] += int(
            candidates.shape[0]
        )

        # --------------------------------------------------------
        # O2M internal NMS
        # --------------------------------------------------------

        if candidates.numel():
            keep = classwise_nms_indices(
                candidates,
                tau_dup,
            )

            candidates = candidates[
                keep
            ]

            candidate_probs = (
                candidate_probs[keep]
            )

        totals[
            "box_dhf_extras"
        ] += int(
            candidates.shape[0]
        )

        # --------------------------------------------------------
        # Mask-aware reliability
        # --------------------------------------------------------

        if candidates.numel():
            stability = mask_stability(
                candidate_probs,
                low=stability_low,
                high=stability_high,
            )

            reliability = torch.sqrt(
                (
                    candidates[:, 4]
                    * stability
                ).clamp_min(0.0)
            )

            reliabilities.extend(
                reliability
                .detach()
                .cpu()
                .tolist()
            )

            reliable = (
                reliability
                >= reliability_threshold
            )

            totals[
                "rejected_reliability"
            ] += int(
                (~reliable)
                .sum()
                .item()
            )

            extras = candidates[
                reliable
            ]

            extra_probs = (
                candidate_probs[
                    reliable
                ]
            )

            extra_masks = (
                extra_probs
                >= mask_threshold
            )

        else:
            extras = candidates

            extra_masks = torch.zeros(
                (
                    0,
                    proto.shape[-2],
                    proto.shape[-1],
                ),
                device=proto.device,
                dtype=torch.bool,
            )

        totals[
            "mask_dhf_extras"
        ] += int(
            extras.shape[0]
        )

        # --------------------------------------------------------
        # Final fusion
        # --------------------------------------------------------

        if (
            anchors.numel()
            and extras.numel()
        ):
            fused = torch.cat(
                [anchors, extras],
                dim=0,
            )

            fused_masks = torch.cat(
                [
                    anchor_masks.float(),
                    extra_masks.float(),
                ],
                dim=0,
            )

        elif anchors.numel():
            fused = anchors
            fused_masks = (
                anchor_masks.float()
            )

        else:
            fused = extras
            fused_masks = (
                extra_masks.float()
            )

        if fused.numel():
            order = torch.argsort(
                fused[:, 4],
                descending=True,
            )

            fused = fused[order]
            fused_masks = (
                fused_masks[order]
            )

        totals["pseudo"] += int(
            fused.shape[0]
        )

        labels_out.append(fused)
        masks_out.append(fused_masks)

    totals["reliabilities"] = (
        reliabilities
    )

    return (
        labels_out,
        masks_out,
        totals,
    )
