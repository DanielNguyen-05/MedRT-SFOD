#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))


from smoke_stage2_mt_seg import (  # noqa: E402
    TargetMTDataset,
    build_pseudo_batch,
    collate,
    list_images,
    resolve_device,
    seed_everything,
    setup_teacher_student,
    update_teacher_ema,
)

from mask_dhf_seg import (  # noqa: E402
    generate_mask_dhf_pseudo_masks,
)

from cc_dhf_seg import (  # noqa: E402
    generate_cc_dhf_pseudo_masks,
)

from boundary_dhf_seg import (  # noqa: E402
    generate_boundary_dhf_pseudo_masks,
)

from durr_seg import (  # noqa: E402
    compute_durr_loss,
    generate_durr_pseudo_masks,
)

from segmard_seg import (
    add_segmard_args,
    compute_segmard_loss,
)

# ================================================================
# MARD
# ================================================================

class SegmentInputFeatureHook:
    """
    Capture P3/P4/P5 features entering Segment26.
    """

    def __init__(
        self,
        model: nn.Module,
    ):
        self.latest = None

        self.head = model.model[-1]

        if not (
            hasattr(self.head, "one2one")
            and hasattr(
                self.head,
                "one2many",
            )
        ):
            raise RuntimeError(
                "Final head is not dual-head"
            )

        self.handle = (
            self.head
            .register_forward_pre_hook(
                self._hook
            )
        )

    @staticmethod
    def _valid_feature_list(x):
        return (
            isinstance(
                x,
                (list, tuple),
            )
            and len(x) >= 3
            and all(
                isinstance(t, torch.Tensor)
                and t.ndim == 4
                for t in x[:3]
            )
        )

    def _hook(
        self,
        _module,
        inputs,
    ):
        if (
            len(inputs) == 1
            and self._valid_feature_list(
                inputs[0]
            )
        ):
            self.latest = list(
                inputs[0][:3]
            )

        elif self._valid_feature_list(
            inputs
        ):
            self.latest = list(
                inputs[:3]
            )

        else:
            self.latest = None

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def variance_loss(
    tokens: torch.Tensor,
    gamma: float,
    eps: float = 1e-4,
):
    if (
        tokens.numel() == 0
        or tokens.shape[0] < 2
    ):
        return tokens.new_zeros(())

    std = torch.sqrt(
        tokens.var(
            dim=0,
            unbiased=False,
        )
        + eps
    )

    return torch.relu(
        gamma - std
    ).mean()


def covariance_loss(
    tokens: torch.Tensor,
    eps: float = 1e-4,
):
    if (
        tokens.numel() == 0
        or tokens.shape[0] < 2
    ):
        return tokens.new_zeros(())

    n, c = tokens.shape

    z = (
        tokens
        - tokens.mean(
            dim=0,
            keepdim=True,
        )
    )

    z = z / (
        z.std(
            dim=0,
            keepdim=True,
        )
        + eps
    )

    cov = (
        z.T @ z
    ) / max(
        n - 1,
        1,
    )

    off_diag = (
        cov
        - torch.diag(
            torch.diagonal(cov)
        )
    )

    return (
        off_diag.pow(2).sum()
        / (
            c * (c - 1)
            + 1e-6
        )
    )


def assign_boxes_to_levels(
    boxes: torch.Tensor,
    stride3: float,
    stride4: float,
    eta: float,
):
    if boxes.numel() == 0:
        return boxes.new_zeros(
            (0,),
            dtype=torch.long,
        )

    sizes = torch.sqrt(
        (
            boxes[:, 2]
            - boxes[:, 0]
        ).clamp(min=1.0)
        *
        (
            boxes[:, 3]
            - boxes[:, 1]
        ).clamp(min=1.0)
    )

    levels = torch.empty_like(
        sizes,
        dtype=torch.long,
    )

    levels[
        sizes
        <= eta * stride3
    ] = 0

    levels[
        (
            sizes
            > eta * stride3
        )
        &
        (
            sizes
            <= eta * stride4
        )
    ] = 1

    levels[
        sizes
        > eta * stride4
    ] = 2

    return levels


def feature_rect(
    box,
    h_f,
    w_f,
    h_pad,
    w_pad,
):
    x1, y1, x2, y2 = (
        box.float()
    )

    x1f = int(
        torch.floor(
            x1 * w_f
            / max(w_pad, 1)
        ).item()
    )

    y1f = int(
        torch.floor(
            y1 * h_f
            / max(h_pad, 1)
        ).item()
    )

    x2f = int(
        torch.ceil(
            x2 * w_f
            / max(w_pad, 1)
        ).item()
    ) - 1

    y2f = int(
        torch.ceil(
            y2 * h_f
            / max(h_pad, 1)
        ).item()
    ) - 1

    x1f = max(
        0,
        min(x1f, w_f - 1),
    )

    x2f = max(
        0,
        min(x2f, w_f - 1),
    )

    y1f = max(
        0,
        min(y1f, h_f - 1),
    )

    y2f = max(
        0,
        min(y2f, h_f - 1),
    )

    if (
        x2f < x1f
        or y2f < y1f
    ):
        return None

    return (
        x1f,
        y1f,
        x2f,
        y2f,
    )


def sample_level_tokens(
    fmap: torch.Tensor,
    boxes: torch.Tensor,
    levels: torch.Tensor,
    target_level: int,
    h_pad: int,
    w_pad: int,
    fg_points: int,
    bg_points: int,
):
    device = fmap.device

    _, h_f, w_f = fmap.shape

    fg_tokens = []

    fg_mask = torch.zeros(
        (h_f, w_f),
        dtype=torch.bool,
        device=device,
    )

    level_mask = (
        levels == target_level
    )

    for box in boxes[level_mask]:
        rect = feature_rect(
            box,
            h_f,
            w_f,
            h_pad,
            w_pad,
        )

        if rect is None:
            continue

        x1, y1, x2, y2 = rect

        xs = torch.randint(
            x1,
            x2 + 1,
            (fg_points,),
            device=device,
        )

        ys = torch.randint(
            y1,
            y2 + 1,
            (fg_points,),
            device=device,
        )

        fg_tokens.append(
            fmap[:, ys, xs].T
        )

        fg_mask[
            y1:y2 + 1,
            x1:x2 + 1,
        ] = True

    bg_coords = (
        ~fg_mask
    ).nonzero(
        as_tuple=False
    )

    if bg_coords.numel():
        n = min(
            bg_points,
            bg_coords.shape[0],
        )

        idx = torch.randint(
            0,
            bg_coords.shape[0],
            (n,),
            device=device,
        )

        sel = bg_coords[idx]

        bg_tokens = fmap[
            :,
            sel[:, 0],
            sel[:, 1],
        ].T

    else:
        ys = torch.randint(
            0,
            h_f,
            (bg_points,),
            device=device,
        )

        xs = torch.randint(
            0,
            w_f,
            (bg_points,),
            device=device,
        )

        bg_tokens = fmap[
            :,
            ys,
            xs,
        ].T

    tokens = (
        fg_tokens
        + [bg_tokens]
    )

    return torch.cat(
        tokens,
        dim=0,
    )


def compute_mard_loss(
    feats,
    pseudo_labels,
    h_pad,
    w_pad,
    args,
):
    total = feats[0].new_zeros(
        ()
    )

    stride3 = (
        float(w_pad)
        / float(
            feats[0].shape[3]
        )
    )

    stride4 = (
        float(w_pad)
        / float(
            feats[1].shape[3]
        )
    )

    stats = {}

    for level_idx, fmap in enumerate(
        feats[:3]
    ):
        tokens_all = []

        for b in range(
            fmap.shape[0]
        ):
            labels = pseudo_labels[b]

            if labels.numel() == 0:
                continue

            boxes = labels[:, :4]

            conf = labels[:, 4]

            keep = (
                conf
                >= args.mard_box_conf
            )

            boxes = boxes[keep]
            conf = conf[keep]

            if boxes.numel() == 0:
                continue

            if (
                boxes.shape[0]
                > args.mard_topk_boxes
            ):
                order = torch.argsort(
                    conf,
                    descending=True,
                )[
                    :args.mard_topk_boxes
                ]

                boxes = boxes[order]

            levels = (
                assign_boxes_to_levels(
                    boxes,
                    stride3,
                    stride4,
                    args.mard_eta,
                )
            )

            tokens = (
                sample_level_tokens(
                    fmap[b],
                    boxes,
                    levels,
                    level_idx,
                    h_pad,
                    w_pad,
                    args.mard_fg_points,
                    args.mard_bg_points,
                )
            )

            if tokens is not None:
                tokens_all.append(
                    tokens
                )

        if tokens_all:
            z = torch.cat(
                tokens_all,
                dim=0,
            )

            var = variance_loss(
                z,
                args.mard_gamma,
            )

            cov = covariance_loss(z)

        else:
            var = fmap.new_zeros(())
            cov = fmap.new_zeros(())

        level_loss = (
            args.mard_alpha
            * var
            +
            args.mard_beta
            * cov
        )

        total = (
            total
            + level_loss
        )

        stats[
            f"p{level_idx + 3}_var"
        ] = float(
            var.detach()
        )

        stats[
            f"p{level_idx + 3}_cov"
        ] = float(
            cov.detach()
        )

    return total, stats


def mard_weight(
    args,
    global_step,
    steps_per_epoch,
    avg_conf,
):
    warmup_steps = max(
        1,
        int(
            args.mard_warmup_epochs
            * steps_per_epoch
        ),
    )

    ramp = min(
        1.0,
        float(global_step)
        / float(warmup_steps),
    )

    gate = (
        avg_conf
        - args.mard_gate_threshold
    ) / max(
        1.0
        - args.mard_gate_threshold,
        1e-6,
    )

    gate = float(
        np.clip(
            gate,
            0.0,
            1.0,
        )
    )

    weight = (
        args.mard_lambda0
        * ramp
        * gate
    )

    return min(
        weight,
        args.mard_lambda_max,
    )


def average_confidence(
    labels,
):
    values = [
        x[:, 4]
        for x in labels
        if x.numel()
    ]

    if not values:
        return 0.0

    return float(
        torch.cat(values)
        .mean()
        .item()
    )



def slice_batch_structure(
    obj,
    indices: torch.Tensor,
    full_batch_size: int,
):
    """
    Recursively slice Student training outputs along batch dimension.

    DURR needs one Student forward over the WHOLE target minibatch so
    teacher-empty images can still receive safe-negative supervision. Native
    SFSeg/SegMARD losses are then computed only on pseudo-valid images.
    """
    if isinstance(obj, torch.Tensor):
        if obj.ndim > 0 and int(obj.shape[0]) == int(full_batch_size):
            return obj.index_select(0, indices)
        return obj

    if isinstance(obj, dict):
        return {
            k: slice_batch_structure(v, indices, full_batch_size)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            slice_batch_structure(v, indices, full_batch_size)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return tuple(
            slice_batch_structure(v, indices, full_batch_size)
            for v in obj
        )

    return obj


def durr_warmup_scale(
    epoch_index: int,
    batch_index: int,
    steps_per_epoch: int,
    warmup_epochs: float,
) -> float:
    if warmup_epochs <= 0:
        return 1.0
    step = epoch_index * max(steps_per_epoch, 1) + batch_index
    total = max(1.0, warmup_epochs * max(steps_per_epoch, 1))
    return float(min(1.0, max(0.0, step / total)))


# ================================================================
# Main
# ================================================================


def inject_soft_instance_masks(
    pseudo_batch: dict,
    soft_masks: list[torch.Tensor],
) -> dict:
    """
    Replace only native instance-mask BCE targets with BDL soft masks.

    build_pseudo_batch() is still called with HARD geometry masks first, so
    boxes/classes/semantic pseudo targets remain exactly on the baseline path.
    """
    if "masks" not in pseudo_batch:
        raise RuntimeError("Pseudo batch has no instance masks")

    target_masks = pseudo_batch["masks"]
    if target_masks.ndim != 3:
        raise RuntimeError(
            f"Expected pseudo_batch['masks'] [N,H,W], got {tuple(target_masks.shape)}"
        )

    target_hw = tuple(target_masks.shape[-2:])
    pieces = []
    expected_n = 0
    cursor = 0

    for masks_i in soft_masks:
        if masks_i.ndim != 3:
            raise RuntimeError(
                f"Expected BDL soft masks [N,H,W], got {tuple(masks_i.shape)}"
            )
        n_i = int(masks_i.shape[0])
        expected_n += n_i
        if n_i == 0:
            continue
        resized = F.interpolate(
            masks_i[:, None].float(),
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )[:, 0].clamp(0.0, 1.0)

        # Do not let bilinear interpolation create support outside the exact
        # HARD baseline masks produced by build_pseudo_batch(). This guarantees
        # that BDL changes only boundary target confidence, not pseudo geometry.
        hard_support = (
            target_masks[cursor:cursor + n_i] > 0.5
        ).to(device=resized.device, dtype=resized.dtype)
        resized = resized * hard_support
        pieces.append(resized)
        cursor += n_i

    if expected_n != int(target_masks.shape[0]):
        raise RuntimeError(
            f"BDL soft-mask count={expected_n} != pseudo batch masks={target_masks.shape[0]}"
        )

    if pieces:
        soft = torch.cat(pieces, dim=0).to(
            device=target_masks.device,
            dtype=target_masks.dtype,
        )
    else:
        soft = target_masks.new_zeros(target_masks.shape)

    if soft.shape != target_masks.shape:
        raise RuntimeError(
            f"BDL resized masks {tuple(soft.shape)} != baseline masks {tuple(target_masks.shape)}"
        )

    out = dict(pseudo_batch)
    out["masks"] = soft
    return out


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\\n")


def save_bdl_debug_panels(
    debug_items,
    paths,
    out_dir: Path,
    epoch: int,
    batch_i: int,
    saved_so_far: int,
    max_images: int,
) -> int:
    """Save label-free BDL diagnostic panels for model understanding."""
    if saved_so_far >= max_images:
        return saved_so_far

    import matplotlib.pyplot as plt

    vis_dir = out_dir / "bdl_debug"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for local_i, item in enumerate(debug_items):
        if saved_so_far >= max_images:
            break

        image = item["image"].float().clamp(0, 1)
        if image.ndim != 3:
            continue
        image_np = image.permute(1, 2, 0).numpy()
        h, w = image_np.shape[:2]

        def up(name):
            x = item[name].float()[None, None]
            return F.interpolate(
                x,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )[0, 0].numpy()

        maps = [
            ("O2O hard", up("o2o_hard"), "gray"),
            ("O2M witnesses", up("o2m_witness"), "viridis"),
            ("Disagreement", up("disagreement"), "magma"),
            ("Inner boundary", up("boundary"), "gray"),
            ("Soft target", up("soft_target"), "viridis"),
        ]

        fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))
        axes[0].imshow(image_np)
        axes[0].set_title("Target image")
        axes[0].axis("off")

        for ax, (title, arr, cmap) in zip(axes[1:], maps):
            im = ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_title(title)
            ax.axis("off")
            if title == "Disagreement":
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        stem = Path(str(paths[local_i])).stem if local_i < len(paths) else f"img{local_i}"
        fig.suptitle(
            f"BDL label-free diagnostic | epoch={epoch} batch={batch_i} | {stem}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(
            vis_dir / f"e{epoch:02d}_b{batch_i:03d}_{stem}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)
        saved_so_far += 1

    return saved_so_far


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--weights",
        required=True,
    )

    ap.add_argument(
        "--target-images",
        required=True,
    )

    ap.add_argument(
        "--out-dir",
        default=(
            "runs/seg/dense_sfseg/"
            "cvc_dense"
        ),
    )

    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )

    ap.add_argument(
        "--batch",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=60,
    )

    ap.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 = all batches",
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
    )

    ap.add_argument(
        "--ema",
        type=float,
        default=0.999,
    )

    # Mask-DHF
    ap.add_argument(
        "--tau-o2o",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--tau-o2m",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--tau-no",
        type=float,
        default=0.2,
    )

    ap.add_argument(
        "--tau-dup",
        type=float,
        default=0.7,
    )

    ap.add_argument(
        "--mask-thr",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--stability-low",
        type=float,
        default=0.40,
    )

    ap.add_argument(
        "--stability-high",
        type=float,
        default=0.60,
    )

    ap.add_argument(
        "--mask-rel-thr",
        type=float,
        default=0.744898,
    )

    ap.add_argument(
        "--min-mask-pixels",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--dhf-mode",
        choices=("mask", "cc", "bdl", "durr"),
        default="mask",
        help=(
            "mask = current Mask-DHF; "
            "cc = rejected simple consensus ablation; "
            "bdl = cross-head Boundary-Disagreement Learning + Mask-DHF coverage; "
            "durr = Dual-head Uncertainty Reliability Routing on top of BDL pseudo labels"
        ),
    )

    ap.add_argument(
        "--cc-tau-match",
        type=float,
        default=0.5,
        help="Same-class box IoU threshold for O2M consensus witnesses.",
    )

    ap.add_argument(
        "--cc-max-witnesses",
        type=int,
        default=5,
        help=(
            "Maximum O2M consensus witnesses per O2O anchor. "
            "0 means unlimited."
        ),
    )

    # Cross-head Boundary Disagreement Learning (BDL)
    ap.add_argument(
        "--bdl-tau-match",
        type=float,
        default=0.5,
        help="Same-class O2M/O2O box IoU threshold for uncertainty witnesses.",
    )
    ap.add_argument(
        "--bdl-max-witnesses",
        type=int,
        default=5,
        help="Maximum O2M uncertainty witnesses per O2O anchor; 0 = unlimited.",
    )
    ap.add_argument(
        "--bdl-boundary-kernel",
        type=int,
        default=3,
        help="Odd erosion kernel defining the inner O2O pseudo-mask boundary.",
    )
    ap.add_argument(
        "--bdl-debug-vis",
        action="store_true",
        help="Save label-free O2O/O2M/disagreement/boundary/soft-target panels.",
    )
    ap.add_argument(
        "--bdl-debug-max-images",
        type=int,
        default=8,
        help="Maximum BDL diagnostic panels saved for a run.",
    )

    # DURR: Dual-head Uncertainty Reliability Routing
    ap.add_argument(
        "--durr-boundary-kernel",
        type=int,
        default=5,
        help="Odd kernel for the two-sided DURR boundary band.",
    )
    ap.add_argument(
        "--durr-direction-min-abs",
        type=float,
        default=0.03,
        help="Minimum absolute signed O2O/O2M disagreement used for routing.",
    )
    ap.add_argument(
        "--durr-direction-margin",
        type=float,
        default=0.05,
        help="Required Student movement along the signed Teacher direction.",
    )
    ap.add_argument(
        "--durr-rescue-conf",
        type=float,
        default=0.80,
        help="Minimum O2M confidence for reliable rescue when native O2O is empty.",
    )
    ap.add_argument(
        "--durr-rescue-stability",
        type=float,
        default=0.80,
        help="Minimum O2M mask threshold-stability for reliable rescue.",
    )
    ap.add_argument(
        "--durr-evidence-conf-floor",
        type=float,
        default=0.10,
        help="Low confidence floor used only to build conservative Teacher evidence maps.",
    )
    ap.add_argument(
        "--durr-safe-bg-teacher-prob",
        type=float,
        default=0.15,
        help="Both Teacher heads must be below this probability to mark safe background.",
    )
    ap.add_argument(
        "--durr-hall-student-prob",
        type=float,
        default=0.70,
        help="Student foreground confidence defining hallucinated pixels.",
    )
    ap.add_argument(
        "--durr-hall-area-thr",
        type=float,
        default=0.10,
        help="Trigger hallucination suppression if high-conf Student FG exceeds this image fraction.",
    )
    ap.add_argument(
        "--durr-lambda-direction",
        type=float,
        default=0.20,
    )
    ap.add_argument(
        "--durr-lambda-rescue",
        type=float,
        default=0.50,
    )
    ap.add_argument(
        "--durr-lambda-hall",
        type=float,
        default=0.20,
    )
    ap.add_argument(
        "--durr-warmup-epochs",
        type=float,
        default=3.0,
        help="Linear DURR loss warmup.",
    )

    # MARD
    ap.add_argument(
        "--mard-lambda0",
        type=float,
        default=0.05,
    )

    ap.add_argument(
        "--mard-lambda-max",
        type=float,
        default=0.2,
    )

    ap.add_argument(
        "--mard-gamma",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--mard-alpha",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--mard-beta",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--mard-warmup-epochs",
        type=float,
        default=5.0,
    )

    ap.add_argument(
        "--mard-gate-threshold",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--mard-topk-boxes",
        type=int,
        default=15,
    )

    ap.add_argument(
        "--mard-fg-points",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--mard-bg-points",
        type=int,
        default=128,
    )

    ap.add_argument(
        "--mard-eta",
        type=float,
        default=12.0,
    )

    ap.add_argument(
        "--mard-box-conf",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--mard-mode",
        choices=("box", "mask"),
        default="mask",
        help=(
            "box = original pseudo-box-guided MARD; "
            "mask = segmentation-guided SegMARD"
        ),
    )

    add_segmard_args(ap)

    ap.add_argument(
        "--device",
        default="0",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=29,
    )

    ap.add_argument(
        "--print-freq",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--save-interval",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--post-trace-gt-masks",
        default=None,
        help=(
            "OPTIONAL evaluation-only GT mask directory. If provided, after the "
            "final checkpoint is frozen the training script automatically runs "
            "Teacher-vs-Student tracing; GT is never read during adaptation."
        ),
    )
    ap.add_argument(
        "--post-trace-max-images",
        type=int,
        default=0,
        help="0 = trace all target images after training; otherwise limit count.",
    )

    args = ap.parse_args()

    seed_everything(
        args.seed
    )

    device = resolve_device(
        args.device
    )

    images = list_images(
        Path(
            args.target_images
        ).resolve()
    )

    dataset = TargetMTDataset(
        images,
        args.imgsz,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
        collate_fn=collate,
    )

    (
        teacher,
        student,
        student_wrapper,
        criterion,
    ) = setup_teacher_student(
        args.weights,
        device,
        epochs=args.epochs,
    )

    optimizer = optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
    )

    scheduler = (
        optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=(
                args.lr * 0.01
            ),
        )
    )

    out_dir = Path(
        args.out_dir
    ).resolve()

    checkpoint_dir = (
        out_dir
        / "checkpoints"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    hook = SegmentInputFeatureHook(
        student
    )

    print("=" * 72)
    print("DENSE MEDRT-SFSEG STAGE-2")
    print("=" * 72)

    print("target images :", len(images))
    print("labels used   : NO")
    print("GT masks used : NO")
    print("epochs        :", args.epochs)
    print("batch         :", args.batch)

    print()
    print(
        "DHF mode      :",
        args.dhf_mode,
    )
    print(
        "Mask-DHF rel  :",
        args.mask_rel_thr,
    )

    print(
        "stability     :",
        args.stability_low,
        args.stability_high,
    )

    if args.dhf_mode == "cc":
        print(
            "CC match IoU :",
            args.cc_tau_match,
        )
        print(
            "CC witnesses :",
            args.cc_max_witnesses,
        )
        print(
            "CC fusion    : branch-balanced consensus",
        )
    elif args.dhf_mode == "bdl":
        print("BDL match IoU:", args.bdl_tau_match)
        print("BDL witnesses:", args.bdl_max_witnesses)
        print("BDL boundary : inner mask band, kernel", args.bdl_boundary_kernel)
        print("BDL signal   : weighted O2O/O2M pixel-wise |P_o2o-P_o2m|")
        print("BDL target   : hard geometry + disagreement-softened inner boundary")
        print("BDL coverage : original Mask-DHF for unmatched novel O2M extras")
    elif args.dhf_mode == "durr":
        print("DURR base    : BDL pseudo labels + Mask-DHF coverage")
        print("DURR signed  : r_m*P_o2m - r_o*P_o2o")
        print("DURR band    : two-sided, kernel", args.durr_boundary_kernel)
        print("DURR rescue  : conf>=", args.durr_rescue_conf, "stab>=", args.durr_rescue_stability)
        print("DURR safe BG : Teacher evidence <", args.durr_safe_bg_teacher_prob)
        print("DURR lambdas :", args.durr_lambda_direction, args.durr_lambda_rescue, args.durr_lambda_hall)
        print("DURR warmup  :", args.durr_warmup_epochs, "epochs")

    print()
    print(
        "MARD lambda0  :",
        args.mard_lambda0,
    )

    print(
        "MARD warmup   :",
        args.mard_warmup_epochs,
    )

    print(
        "MARD mode     :",
        args.mard_mode,
    )

    if args.mard_mode == "box":
        print(
            "FG definition : inside pseudo bounding box"
        )
        print(
            "BG definition : outside pseudo bounding boxes"
        )
    else:
        print(
            "FG definition : inside Teacher pseudo segmentation mask"
        )
        print(
            "FG erosion    :",
            args.segmard_erode_kernel,
        )
        print(
            "mask dilation :",
            args.segmard_dilate_kernel,
        )
        print(
            "hard BG ratio :",
            args.segmard_hard_bg_ratio,
        )
        print(
            "reliability wt:",
            "ON"
            if args.segmard_reliability_weighting
            else "OFF",
        )

    print(
        "var/cov obj.   : "
        + (
            "reliability-weighted"
            if (
                args.mard_mode == "mask"
                and args.segmard_reliability_weighting
            )
            else "unchanged"
        )
    )

    global_step = 0
    bdl_debug_saved = 0
    final_ckpt = None
    bdl_history = out_dir / "bdl_diagnostics.jsonl"
    durr_history = out_dir / "durr_diagnostics.jsonl"
    if args.dhf_mode == "bdl" and bdl_history.exists():
        bdl_history.unlink()
    if args.dhf_mode == "durr" and durr_history.exists():
        durr_history.unlink()

    try:
        for epoch in range(
            args.epochs
        ):
            student.train()

            successful = 0
            skipped = 0

            totals = {
                "loss": 0.0,
                "seg_loss": 0.0,
                "semseg_loss": 0.0,
                "mard": 0.0,
                "lambda": 0.0,
                "pseudo": 0,
                "anchors": 0,
                "box_extras": 0,
                "mask_extras": 0,
                "rejected_rel": 0,
                "consensus_anchors": 0,
                "consensus_witnesses": 0,
                "consensus_fallbacks": 0,
                "consensus_shift_sum": 0.0,
                "consensus_shift_count": 0,
                "consensus_mask_iou_sum": 0.0,
                "consensus_mask_iou_count": 0,
                "bdl_anchors": 0,
                "bdl_witnesses": 0,
                "bdl_boundary_fallbacks": 0,
                "bdl_boundary_disagreement_sum": 0.0,
                "bdl_boundary_disagreement_count": 0,
                "bdl_interior_disagreement_sum": 0.0,
                "bdl_interior_disagreement_count": 0,
                "bdl_soft_target_shift_sum": 0.0,
                "bdl_soft_target_shift_count": 0,
                "bdl_boundary_pixels": 0,
                "bdl_interior_pixels": 0,
                "durr_loss": 0.0,
                "durr_dir_loss": 0.0,
                "durr_rescue_loss": 0.0,
                "durr_hall_loss": 0.0,
                "durr_direction_pixels": 0.0,
                "durr_rescue_pixels": 0.0,
                "durr_hall_pixels": 0.0,
                "durr_hall_trigger_images": 0.0,
                "durr_teacher_empty_images": 0,
                "durr_rescue_images": 0,
                "durr_rescue_instances": 0,
                "fg_tokens": 0.0,
                "hard_bg_tokens": 0.0,
                "easy_bg_tokens": 0.0,
                "core_fallbacks": 0.0,
                "rel_sum": 0.0,
                "rel_count": 0,
                "rel_min": 1.0,
                "rel_max": 0.0,
                "token_weight_sum": 0.0,
            }

            start = time.time()

            for batch_i, (
                weak,
                strong,
                _paths,
            ) in enumerate(
                loader,
                start=1,
            ):
                if (
                    args.max_batches > 0
                    and batch_i
                    > args.max_batches
                ):
                    break

                weak = weak.to(
                    device,
                    non_blocking=True,
                )

                strong = strong.to(
                    device,
                    non_blocking=True,
                )

                bdl_debug_items = None
                durr_routes = None

                if args.dhf_mode == "mask":
                    (
                        labels,
                        geometry_masks,
                        instance_reliabilities,
                        ps,
                    ) = generate_mask_dhf_pseudo_masks(
                        teacher,
                        weak,
                        tau_o2o=args.tau_o2o,
                        tau_o2m=args.tau_o2m,
                        tau_no=args.tau_no,
                        tau_dup=args.tau_dup,
                        mask_threshold=args.mask_thr,
                        stability_low=args.stability_low,
                        stability_high=args.stability_high,
                        reliability_threshold=args.mask_rel_thr,
                        min_mask_pixels=args.min_mask_pixels,
                        return_instance_reliability=True,
                    )
                    supervision_masks = geometry_masks

                elif args.dhf_mode == "cc":
                    (
                        labels,
                        geometry_masks,
                        instance_reliabilities,
                        ps,
                    ) = generate_cc_dhf_pseudo_masks(
                        teacher,
                        weak,
                        tau_o2o=args.tau_o2o,
                        tau_o2m=args.tau_o2m,
                        tau_no=args.tau_no,
                        tau_dup=args.tau_dup,
                        tau_match=args.cc_tau_match,
                        max_consensus_witnesses=args.cc_max_witnesses,
                        mask_threshold=args.mask_thr,
                        stability_low=args.stability_low,
                        stability_high=args.stability_high,
                        reliability_threshold=args.mask_rel_thr,
                        min_mask_pixels=args.min_mask_pixels,
                        return_instance_reliability=True,
                    )
                    supervision_masks = geometry_masks

                elif args.dhf_mode == "bdl":
                    bdl_result = generate_boundary_dhf_pseudo_masks(
                        teacher,
                        weak,
                        tau_o2o=args.tau_o2o,
                        tau_o2m=args.tau_o2m,
                        tau_no=args.tau_no,
                        tau_dup=args.tau_dup,
                        tau_match=args.bdl_tau_match,
                        max_witnesses=args.bdl_max_witnesses,
                        mask_threshold=args.mask_thr,
                        stability_low=args.stability_low,
                        stability_high=args.stability_high,
                        reliability_threshold=args.mask_rel_thr,
                        min_mask_pixels=args.min_mask_pixels,
                        boundary_kernel=args.bdl_boundary_kernel,
                        return_debug=(
                            args.bdl_debug_vis
                            and bdl_debug_saved < args.bdl_debug_max_images
                        ),
                    )
                    if len(bdl_result) == 6:
                        (
                            labels,
                            supervision_masks,
                            geometry_masks,
                            instance_reliabilities,
                            ps,
                            bdl_debug_items,
                        ) = bdl_result
                    else:
                        (
                            labels,
                            supervision_masks,
                            geometry_masks,
                            instance_reliabilities,
                            ps,
                        ) = bdl_result
                    durr_routes = None

                else:
                    (
                        labels,
                        supervision_masks,
                        geometry_masks,
                        instance_reliabilities,
                        ps,
                        durr_routes,
                    ) = generate_durr_pseudo_masks(
                        teacher=teacher,
                        weak_imgs=weak,
                        tau_o2o=args.tau_o2o,
                        tau_o2m=args.tau_o2m,
                        tau_no=args.tau_no,
                        tau_dup=args.tau_dup,
                        tau_match=args.bdl_tau_match,
                        max_witnesses=args.bdl_max_witnesses,
                        mask_threshold=args.mask_thr,
                        stability_low=args.stability_low,
                        stability_high=args.stability_high,
                        reliability_threshold=args.mask_rel_thr,
                        min_mask_pixels=args.min_mask_pixels,
                        bdl_boundary_kernel=args.bdl_boundary_kernel,
                        durr_boundary_kernel=args.durr_boundary_kernel,
                        durr_direction_min_abs=args.durr_direction_min_abs,
                        durr_rescue_conf=args.durr_rescue_conf,
                        durr_rescue_stability=args.durr_rescue_stability,
                        durr_evidence_conf_floor=args.durr_evidence_conf_floor,
                        durr_safe_bg_teacher_prob=args.durr_safe_bg_teacher_prob,
                    )

                if not (
                    len(labels)
                    == len(supervision_masks)
                    == len(geometry_masks)
                    == len(instance_reliabilities)
                ):
                    raise RuntimeError(
                        "DHF/BDL batch output lengths are misaligned"
                    )

                for bi_align in range(len(labels)):
                    n_lab = int(labels[bi_align].shape[0])
                    n_sup = int(supervision_masks[bi_align].shape[0])
                    n_geo = int(geometry_masks[bi_align].shape[0])
                    n_rel = int(instance_reliabilities[bi_align].shape[0])
                    if not (n_lab == n_sup == n_geo == n_rel):
                        raise RuntimeError(
                            f"DHF/BDL image {bi_align}: labels={n_lab} "
                            f"supervision={n_sup} geometry={n_geo} rel={n_rel}"
                        )

                valid = [
                    i
                    for i, x
                    in enumerate(labels)
                    if x.shape[0] > 0
                ]

                # Non-DURR baselines keep the original behavior: teacher-empty
                # batches do not update the Student.
                if not valid and args.dhf_mode != "durr":
                    skipped += 1
                    global_step += 1
                    continue

                # Lists aligned to pseudo-valid images. They may be empty in
                # DURR mode; teacher-empty images can still contribute L_hall.
                strong_valid = strong[valid] if valid else strong[:0]
                labels_valid = [labels[i] for i in valid]
                supervision_masks_valid = [
                    supervision_masks[i] for i in valid
                ]
                geometry_masks_valid = [
                    geometry_masks[i] for i in valid
                ]
                reliabilities_valid = [
                    instance_reliabilities[i] for i in valid
                ]

                if reliabilities_valid:
                    rel_cat = torch.cat(
                        [
                            r.detach().float().flatten()
                            for r in reliabilities_valid
                            if r.numel()
                        ],
                        dim=0,
                    )
                else:
                    rel_cat = strong.new_zeros((0,))

                if rel_cat.numel():
                    batch_rel_mean = float(rel_cat.mean().item())
                    batch_rel_min = float(rel_cat.min().item())
                    batch_rel_max = float(rel_cat.max().item())
                else:
                    batch_rel_mean = 0.0
                    batch_rel_min = 0.0
                    batch_rel_max = 0.0

                if (
                    args.dhf_mode == "bdl"
                    and args.bdl_debug_vis
                    and bdl_debug_items is not None
                    and valid
                ):
                    valid_debug = [
                        bdl_debug_items[i]
                        for i in valid
                    ]
                    valid_paths = [
                        _paths[i]
                        for i in valid
                    ]
                    bdl_debug_saved = save_bdl_debug_panels(
                        valid_debug,
                        valid_paths,
                        out_dir,
                        epoch + 1,
                        batch_i,
                        bdl_debug_saved,
                        args.bdl_debug_max_images,
                    )

                avg_conf = (
                    average_confidence(labels_valid)
                    if labels_valid
                    else 0.0
                )

                hook.latest = None

                if args.dhf_mode == "durr":
                    # ONE Student forward over the full batch. This is critical:
                    # teacher-empty images must remain visible to DURR.
                    student_outputs_all = student(strong)
                    feats_all = hook.latest

                    if feats_all is None:
                        raise RuntimeError(
                            "DURR/MARD feature hook captured no features"
                        )

                    warmup_scale = durr_warmup_scale(
                        epoch,
                        batch_i - 1,
                        len(loader),
                        args.durr_warmup_epochs,
                    )
                    durr_loss, durr_stats = compute_durr_loss(
                        student_outputs_all,
                        durr_routes,
                        direction_margin=args.durr_direction_margin,
                        lambda_direction=args.durr_lambda_direction,
                        lambda_rescue=args.durr_lambda_rescue,
                        lambda_hallucination=args.durr_lambda_hall,
                        hall_student_prob=args.durr_hall_student_prob,
                        hall_area_threshold=args.durr_hall_area_thr,
                        warmup_scale=warmup_scale,
                    )

                    if valid:
                        idx_tensor = torch.tensor(
                            valid,
                            device=strong.device,
                            dtype=torch.long,
                        )
                        student_outputs = slice_batch_structure(
                            student_outputs_all,
                            idx_tensor,
                            int(strong.shape[0]),
                        )
                        feats = [
                            f.index_select(0, idx_tensor)
                            for f in feats_all
                        ]
                    else:
                        student_outputs = None
                        feats = None
                else:
                    student_outputs = student(strong_valid)
                    feats = hook.latest
                    durr_loss = strong_valid.sum() * 0.0
                    durr_stats = {
                        "durr_loss": 0.0,
                        "durr_dir_loss": 0.0,
                        "durr_rescue_loss": 0.0,
                        "durr_hall_loss": 0.0,
                        "durr_direction_pixels_student": 0.0,
                        "durr_rescue_pixels_student": 0.0,
                        "durr_hall_pixels_student": 0.0,
                        "durr_hall_trigger_images": 0.0,
                    }

                    if feats is None:
                        raise RuntimeError(
                            "MARD feature hook captured no features"
                        )

                if valid:
                    # Always construct boxes/classes/semantic targets from HARD geometry.
                    pseudo_batch = build_pseudo_batch(
                        labels_valid,
                        geometry_masks_valid,
                        strong_valid.shape,
                    )

                    # BDL and DURR keep BDL's soft instance-mask targets.
                    if args.dhf_mode in ("bdl", "durr"):
                        pseudo_batch = inject_soft_instance_masks(
                            pseudo_batch,
                            supervision_masks_valid,
                        )

                    det_vec, loss_items = criterion(
                        student_outputs,
                        pseudo_batch,
                    )
                    sfseg_loss = det_vec.sum()

                    if args.mard_mode == "box":
                        mard_loss, mard_stats = compute_mard_loss(
                            feats,
                            labels_valid,
                            int(strong_valid.shape[2]),
                            int(strong_valid.shape[3]),
                            args,
                        )
                    else:
                        mard_loss, mard_stats = compute_segmard_loss(
                            feats=feats,
                            pseudo_labels=labels_valid,
                            pseudo_masks=geometry_masks_valid,
                            pseudo_reliabilities=(
                                reliabilities_valid
                                if args.segmard_reliability_weighting
                                else None
                            ),
                            h_pad=int(strong_valid.shape[2]),
                            w_pad=int(strong_valid.shape[3]),
                            args=args,
                        )

                    lambda_mard = mard_weight(
                        args,
                        global_step,
                        len(loader),
                        avg_conf,
                    )
                else:
                    # DURR-only update on teacher-empty images.
                    sfseg_loss = durr_loss * 0.0
                    mard_loss = durr_loss * 0.0
                    lambda_mard = 0.0
                    loss_items = torch.zeros(
                        5,
                        device=strong.device,
                        dtype=strong.dtype,
                    )
                    mard_stats = {}

                total_loss = (
                    sfseg_loss
                    + lambda_mard * mard_loss
                    + durr_loss
                )

                # If a DURR-only teacher-empty batch produced no active route,
                # preserve the original skip behavior rather than taking a
                # meaningless zero-gradient optimizer step.
                if (
                    not valid
                    and args.dhf_mode == "durr"
                    and float(durr_loss.detach().abs().item()) <= 1e-12
                ):
                    skipped += 1
                    global_step += 1
                    continue

                if not torch.isfinite(
                    total_loss
                ):
                    raise RuntimeError(
                        "Non-finite total loss"
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

                total_loss.backward()

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        student.parameters(),
                        args.grad_clip,
                    )
                )

                if not torch.isfinite(
                    torch.as_tensor(
                        grad_norm
                    )
                ):
                    raise RuntimeError(
                        "Non-finite gradient"
                    )

                optimizer.step()

                global_step += 1
                successful += 1

                pseudo_n = sum(
                    x.shape[0]
                    for x in labels_valid
                )

                totals["loss"] += float(
                    total_loss.detach()
                )

                totals["seg_loss"] += float(
                    loss_items[1]
                    .detach()
                )

                totals["semseg_loss"] += float(
                    loss_items[4]
                    .detach()
                )

                totals["mard"] += float(
                    mard_loss.detach()
                )

                totals["lambda"] += (
                    lambda_mard
                )

                totals["pseudo"] += (
                    pseudo_n
                )

                totals["anchors"] += (
                    ps["anchors"]
                )

                totals["box_extras"] += (
                    ps[
                        "box_dhf_extras"
                    ]
                )

                totals["mask_extras"] += (
                    ps[
                        "mask_dhf_extras"
                    ]
                )

                totals["rejected_rel"] += (
                    ps[
                        "rejected_reliability"
                    ]
                )

                if args.dhf_mode == "cc":
                    totals["consensus_anchors"] += int(
                        ps.get("consensus_anchors", 0)
                    )
                    totals["consensus_witnesses"] += int(
                        ps.get("consensus_witnesses", 0)
                    )
                    totals["consensus_fallbacks"] += int(
                        ps.get("consensus_fallbacks", 0)
                    )
                    shift_count = int(
                        ps.get("consensus_abs_shift_count", 0)
                    )
                    totals["consensus_shift_sum"] += float(
                        ps.get("consensus_abs_shift_sum", 0.0)
                    )
                    totals["consensus_shift_count"] += shift_count
                    totals["consensus_mask_iou_sum"] += float(
                        ps.get("consensus_mask_iou_sum", 0.0)
                    )
                    totals["consensus_mask_iou_count"] += int(
                        ps.get("consensus_mask_iou_count", 0)
                    )

                if args.dhf_mode == "bdl":
                    for key in (
                        "bdl_anchors",
                        "bdl_witnesses",
                        "bdl_boundary_fallbacks",
                        "bdl_boundary_disagreement_count",
                        "bdl_interior_disagreement_count",
                        "bdl_soft_target_shift_count",
                        "bdl_boundary_pixels",
                        "bdl_interior_pixels",
                    ):
                        totals[key] += int(ps.get(key, 0))
                    for key in (
                        "bdl_boundary_disagreement_sum",
                        "bdl_interior_disagreement_sum",
                        "bdl_soft_target_shift_sum",
                    ):
                        totals[key] += float(ps.get(key, 0.0))

                    append_jsonl(
                        bdl_history,
                        {
                            "epoch": epoch + 1,
                            "batch": batch_i,
                            "pseudo": pseudo_n,
                            "anchors": int(ps.get("anchors", 0)),
                            "bdl_anchors": int(ps.get("bdl_anchors", 0)),
                            "bdl_witnesses": int(ps.get("bdl_witnesses", 0)),
                            "boundary_disagreement": float(ps.get("bdl_boundary_disagreement_mean", 0.0)),
                            "interior_disagreement": float(ps.get("bdl_interior_disagreement_mean", 0.0)),
                            "boundary_interior_ratio": float(ps.get("bdl_boundary_interior_ratio", 0.0)),
                            "soft_target_shift": float(ps.get("bdl_soft_target_shift_mean", 0.0)),
                            "boundary_pixels": int(ps.get("bdl_boundary_pixels", 0)),
                            "interior_pixels": int(ps.get("bdl_interior_pixels", 0)),
                            "boundary_fallbacks": int(ps.get("bdl_boundary_fallbacks", 0)),
                        },
                    )

                if args.dhf_mode == "durr":
                    totals["durr_loss"] += float(
                        durr_stats.get("durr_loss", 0.0)
                    )
                    totals["durr_dir_loss"] += float(
                        durr_stats.get("durr_dir_loss", 0.0)
                    )
                    totals["durr_rescue_loss"] += float(
                        durr_stats.get("durr_rescue_loss", 0.0)
                    )
                    totals["durr_hall_loss"] += float(
                        durr_stats.get("durr_hall_loss", 0.0)
                    )
                    totals["durr_direction_pixels"] += float(
                        durr_stats.get("durr_direction_pixels_student", 0.0)
                    )
                    totals["durr_rescue_pixels"] += float(
                        durr_stats.get("durr_rescue_pixels_student", 0.0)
                    )
                    totals["durr_hall_pixels"] += float(
                        durr_stats.get("durr_hall_pixels_student", 0.0)
                    )
                    totals["durr_hall_trigger_images"] += float(
                        durr_stats.get("durr_hall_trigger_images", 0.0)
                    )
                    totals["durr_teacher_empty_images"] += int(
                        ps.get("durr_teacher_empty_images", 0)
                    )
                    totals["durr_rescue_images"] += int(
                        ps.get("durr_rescue_images", 0)
                    )
                    totals["durr_rescue_instances"] += int(
                        ps.get("durr_rescue_instances", 0)
                    )

                    append_jsonl(
                        durr_history,
                        {
                            "epoch": epoch + 1,
                            "batch": batch_i,
                            "pseudo": pseudo_n,
                            "anchors": int(ps.get("anchors", 0)),
                            "teacher_empty_images": int(
                                ps.get("durr_teacher_empty_images", 0)
                            ),
                            "rescue_images": int(
                                ps.get("durr_rescue_images", 0)
                            ),
                            "rescue_instances": int(
                                ps.get("durr_rescue_instances", 0)
                            ),
                            "signed_abs_mean": float(
                                ps.get("durr_signed_abs_mean", 0.0)
                            ),
                            "direction_pixels": int(
                                ps.get("durr_direction_pixels", 0)
                            ),
                            "loss": float(
                                durr_stats.get("durr_loss", 0.0)
                            ),
                            "dir_loss": float(
                                durr_stats.get("durr_dir_loss", 0.0)
                            ),
                            "rescue_loss": float(
                                durr_stats.get("durr_rescue_loss", 0.0)
                            ),
                            "hall_loss": float(
                                durr_stats.get("durr_hall_loss", 0.0)
                            ),
                            "hall_trigger_images": int(
                                durr_stats.get("durr_hall_trigger_images", 0.0)
                            ),
                            "warmup": float(
                                durr_stats.get("durr_warmup_scale", 0.0)
                            ),
                        },
                    )

                if rel_cat.numel():
                    totals["rel_sum"] += float(rel_cat.sum().item())
                    totals["rel_count"] += int(rel_cat.numel())
                    totals["rel_min"] = min(
                        totals["rel_min"],
                        batch_rel_min,
                    )
                    totals["rel_max"] = max(
                        totals["rel_max"],
                        batch_rel_max,
                    )

                if args.mard_mode == "mask":
                    for key in (
                        "fg_tokens",
                        "hard_bg_tokens",
                        "easy_bg_tokens",
                        "core_fallbacks",
                    ):
                        totals[key] += float(
                            mard_stats.get(
                                key,
                                0.0,
                            )
                        )

                    totals["token_weight_sum"] += float(
                        mard_stats.get(
                            "token_weight_mean",
                            1.0,
                        )
                    )

                if (
                    batch_i == 1
                    or batch_i
                    % args.print_freq == 0
                    or batch_i == len(loader)
                ):
                    print(
                        f"[E{epoch+1:02d}] "
                        f"batch={batch_i:03d}/{len(loader):03d} "
                        f"pseudo={pseudo_n} "
                        f"loss={float(total_loss.detach()):.4f} "
                        f"seg={float(loss_items[1]):.4f} "
                        f"semseg={float(loss_items[4]):.4f} "
                        f"mard={float(mard_loss.detach()):.4f} "
                        f"lambda={lambda_mard:.6f} "
                        f"box_extra={ps['box_dhf_extras']} "
                        f"mask_extra={ps['mask_dhf_extras']} "
                        + (
                            (
                                f"CC={int(ps.get('consensus_anchors', 0))}/"
                                f"{int(ps.get('consensus_witnesses', 0))} "
                                f"cc_shift="
                                f"{float(ps.get('consensus_abs_shift_mean', 0.0)):.5f} "
                                f"cc_iou="
                                f"{float(ps.get('consensus_mask_iou_mean', 0.0)):.5f} "
                            )
                            if args.dhf_mode == "cc"
                            else ""
                        )
                        + (
                            (
                                f"BDL={int(ps.get('bdl_anchors', 0))}/"
                                f"{int(ps.get('bdl_witnesses', 0))} "
                                f"Dbd/Din="
                                f"{float(ps.get('bdl_boundary_disagreement_mean', 0.0)):.5f}/"
                                f"{float(ps.get('bdl_interior_disagreement_mean', 0.0)):.5f} "
                                f"ratio={float(ps.get('bdl_boundary_interior_ratio', 0.0)):.2f} "
                                f"soft_shift={float(ps.get('bdl_soft_target_shift_mean', 0.0)):.5f} "
                            )
                            if args.dhf_mode == "bdl"
                            else ""
                        )
                        + (
                            (
                                f"DURR="
                                f"{float(durr_stats.get('durr_loss', 0.0)):.4f} "
                                f"(dir={float(durr_stats.get('durr_dir_loss', 0.0)):.4f},"
                                f"res={float(durr_stats.get('durr_rescue_loss', 0.0)):.4f},"
                                f"hall={float(durr_stats.get('durr_hall_loss', 0.0)):.4f}) "
                                f"route_px={int(durr_stats.get('durr_direction_pixels_student', 0))} "
                                f"rescue_px={int(durr_stats.get('durr_rescue_pixels_student', 0))} "
                                f"hall_px={int(durr_stats.get('durr_hall_pixels_student', 0))} "
                            )
                            if args.dhf_mode == "durr"
                            else ""
                        )
                        + (
                            (
                                f"FG/HBG/EBG="
                                f"{int(mard_stats.get('fg_tokens', 0))}/"
                                f"{int(mard_stats.get('hard_bg_tokens', 0))}/"
                                f"{int(mard_stats.get('easy_bg_tokens', 0))} "
                                f"fallback="
                                f"{int(mard_stats.get('core_fallbacks', 0))} "
                                f"rel={batch_rel_mean:.3f}/"
                                f"{batch_rel_min:.3f}/"
                                f"{batch_rel_max:.3f} "
                                f"wmean="
                                f"{float(mard_stats.get('token_weight_mean', 1.0)):.3f} "
                            )
                            if args.mard_mode == "mask"
                            else ""
                        )
                        + f"grad_preclip={float(grad_norm):.2f}",
                        flush=True,
                    )

            if successful == 0:
                raise RuntimeError(
                    "Epoch had no valid batches"
                )

            scheduler.step()

            if hasattr(
                criterion,
                "update",
            ):
                criterion.update()

            # Epoch-level EMA only.
            update_teacher_ema(
                teacher,
                student,
                args.ema,
            )

            denom = max(
                successful,
                1,
            )

            print()
            print(
                f"[Epoch {epoch+1:02d}] DONE "
                f"time={time.time()-start:.1f}s "
                f"valid={successful} "
                f"skipped={skipped} "
                f"pseudo={totals['pseudo']} "
                f"anchors={totals['anchors']} "
                f"box_extra={totals['box_extras']} "
                f"mask_extra={totals['mask_extras']} "
                f"reject_rel={totals['rejected_rel']} "
                + (
                    (
                        f"CC={totals['consensus_anchors']}/"
                        f"{totals['consensus_witnesses']} "
                        f"cc_fallback={totals['consensus_fallbacks']} "
                        f"cc_shift="
                        f"{totals['consensus_shift_sum']/max(totals['consensus_shift_count'], 1):.5f} "
                        f"cc_iou="
                        f"{totals['consensus_mask_iou_sum']/max(totals['consensus_mask_iou_count'], 1):.5f} "
                    )
                    if args.dhf_mode == "cc"
                    else ""
                )
                + (
                    (
                        f"BDL={totals['bdl_anchors']}/{totals['bdl_witnesses']} "
                        f"bdl_fallback={totals['bdl_boundary_fallbacks']} "
                        f"Dbd/Din="
                        f"{totals['bdl_boundary_disagreement_sum']/max(totals['bdl_boundary_disagreement_count'], 1):.5f}/"
                        f"{totals['bdl_interior_disagreement_sum']/max(totals['bdl_interior_disagreement_count'], 1):.5f} "
                        f"ratio="
                        f"{(totals['bdl_boundary_disagreement_sum']/max(totals['bdl_boundary_disagreement_count'], 1))/max(totals['bdl_interior_disagreement_sum']/max(totals['bdl_interior_disagreement_count'], 1), 1e-8):.2f} "
                        f"soft_shift="
                        f"{totals['bdl_soft_target_shift_sum']/max(totals['bdl_soft_target_shift_count'], 1):.5f} "
                    )
                    if args.dhf_mode == "bdl"
                    else ""
                )
                + (
                    (
                        f"DURR={totals['durr_loss']/denom:.4f} "
                        f"(dir={totals['durr_dir_loss']/denom:.4f},"
                        f"res={totals['durr_rescue_loss']/denom:.4f},"
                        f"hall={totals['durr_hall_loss']/denom:.4f}) "
                        f"empty={totals['durr_teacher_empty_images']} "
                        f"rescue={totals['durr_rescue_images']}/"
                        f"{totals['durr_rescue_instances']} "
                        f"hall_trigger={int(totals['durr_hall_trigger_images'])} "
                    )
                    if args.dhf_mode == "durr"
                    else ""
                )
                + f"loss={totals['loss']/denom:.4f} "
                f"seg={totals['seg_loss']/denom:.4f} "
                f"semseg={totals['semseg_loss']/denom:.4f} "
                f"mard={totals['mard']/denom:.4f} "
                f"lambda_avg={totals['lambda']/denom:.6f}"
                + (
                    (
                        f" FG/HBG/EBG="
                        f"{totals['fg_tokens']/denom:.1f}/"
                        f"{totals['hard_bg_tokens']/denom:.1f}/"
                        f"{totals['easy_bg_tokens']/denom:.1f} "
                        f"rel="
                        f"{totals['rel_sum']/max(totals['rel_count'], 1):.3f}/"
                        f"{totals['rel_min']:.3f}/"
                        f"{totals['rel_max']:.3f} "
                        f"wmean="
                        f"{totals['token_weight_sum']/denom:.3f}"
                    )
                    if args.mard_mode == "mask"
                    else ""
                ),
                flush=True,
            )

            save_this_epoch = (
                (epoch + 1) % args.save_interval == 0
                or (epoch + 1) == args.epochs
            )

            if save_this_epoch:
                # Detach hook before serialization.
                hook.close()

                student.eval()

                student_wrapper.model = student

                ckpt = (
                    checkpoint_dir
                    / (
                        f"dense_sfseg_"
                        f"epoch_{epoch+1}.pt"
                    )
                )

                student_wrapper.save(
                    str(ckpt)
                )
                final_ckpt = ckpt

                print(
                    "[SAVE]",
                    ckpt,
                    flush=True,
                )

                if epoch + 1 < args.epochs:
                    student.train()

                    hook = SegmentInputFeatureHook(
                        student
                    )

    finally:
        if (
            hook is not None
            and hook.handle is not None
        ):
            hook.close()

    metadata = {
        "source_free": True,
        "target_labels_used": False,
        "target_gt_masks_used": False,

        "initial_weights":
            str(
                Path(
                    args.weights
                ).resolve()
            ),

        "epochs": args.epochs,

        "dhf_mode": args.dhf_mode,

        "cc_dhf": {
            "enabled": args.dhf_mode == "cc",
            "tau_match": args.cc_tau_match,
            "max_consensus_witnesses": args.cc_max_witnesses,
            "consensus_fusion": (
                "branch-balanced reliability-weighted O2O/O2M soft-mask consensus"
            ),
            "coverage_definition": (
                "same-class best O2O box IoU <= tau_no, then original Mask-DHF NMS+reliability"
            ),
            "o2o_box_policy": "unchanged",
            "o2o_reliability_policy": "unchanged; consensus refines masks only",
        },

        "boundary_disagreement_learning": {
            "enabled": args.dhf_mode == "bdl",
            "tau_match": args.bdl_tau_match,
            "max_witnesses": args.bdl_max_witnesses,
            "boundary_kernel": args.bdl_boundary_kernel,
            "uncertainty": (
                "reliability-weighted mean absolute O2O/O2M soft-mask disagreement"
            ),
            "pseudo_target_rule": (
                "keep O2O hard geometry; soften only the inner positive boundary toward "
                "O2O soft probability by pixel-wise disagreement"
            ),
            "coverage": (
                "unmatched novel O2M follows original Mask-DHF NMS + reliability gate"
            ),
            "target_gt_used": False,
            "debug_visualization": args.bdl_debug_vis,
        },

        "durr": {
            "enabled": args.dhf_mode == "durr",
            "name": "Dual-head Uncertainty Reliability Routing",
            "signed_direction": "r_m * P_o2m - r_o * P_o2o",
            "boundary_kernel": args.durr_boundary_kernel,
            "direction_min_abs": args.durr_direction_min_abs,
            "direction_margin": args.durr_direction_margin,
            "rescue_conf": args.durr_rescue_conf,
            "rescue_stability": args.durr_rescue_stability,
            "safe_bg_teacher_prob": args.durr_safe_bg_teacher_prob,
            "hall_student_prob": args.durr_hall_student_prob,
            "hall_area_threshold": args.durr_hall_area_thr,
            "lambda_direction": args.durr_lambda_direction,
            "lambda_rescue": args.durr_lambda_rescue,
            "lambda_hallucination": args.durr_lambda_hall,
            "warmup_epochs": args.durr_warmup_epochs,
            "target_gt_used_during_adaptation": False,
        },

        "mask_dhf": {
            "tau_o2o":
                args.tau_o2o,
            "tau_o2m":
                args.tau_o2m,
            "tau_no":
                args.tau_no,
            "tau_dup":
                args.tau_dup,
            "stability_low":
                args.stability_low,
            "stability_high":
                args.stability_high,
            "reliability_threshold":
                args.mask_rel_thr,
            "threshold_source":
                "Q25 of initial AdaBN Teacher "
                "Box-DHF extras; label-free",
        },

        "mard": {
            "lambda0":
                args.mard_lambda0,
            "lambda_max":
                args.mard_lambda_max,
            "warmup_epochs":
                args.mard_warmup_epochs,
            "gamma":
                args.mard_gamma,
            "alpha":
                args.mard_alpha,
            "beta":
                args.mard_beta,
            "fg_points":
                args.mard_fg_points,
            "bg_points":
                args.mard_bg_points,
            "eta":
                args.mard_eta,
            "mode":
                args.mard_mode,
            "foreground_definition":
                (
                    "pseudo segmentation mask"
                    if args.mard_mode == "mask"
                    else "pseudo bounding box"
                ),
            "segmard_erode_kernel":
                args.segmard_erode_kernel,
            "segmard_dilate_kernel":
                args.segmard_dilate_kernel,
            "segmard_hard_bg_ratio":
                args.segmard_hard_bg_ratio,
            "segmard_reliability_weighting":
                args.segmard_reliability_weighting,
            "segmard_reliability_definition":
                (
                    "sqrt(box_confidence * threshold_stability)"
                    if args.segmard_reliability_weighting
                    else "disabled"
                ),
            "segmard_reliability_token_policy":
                (
                    "FG=r_i; HBG=max r_i of covering hard regions; EBG=1"
                    if args.segmard_reliability_weighting
                    else "uniform"
                ),
        },

        "ema_momentum":
            args.ema,

        "ema_frequency":
            "epoch",
    }

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out_dir
        / "stage2_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # POST-TRAINING ONLY Teacher-vs-Student trace.
    # This is intentionally after final checkpoint serialization and metadata.
    # Target GT never participates in adaptation or model selection.
    # ---------------------------------------------------------------
    if args.post_trace_gt_masks:
        if final_ckpt is None:
            raise RuntimeError(
                "Post-trace requested but no final Student checkpoint was saved"
            )

        trace_script = SCRIPT_DIR / "trace_teacher_student_masks.py"
        if not trace_script.exists():
            raise FileNotFoundError(
                f"Automatic post-trace requires {trace_script}"
            )

        trace_out = out_dir / "final_teacher_student_trace"
        cmd = [
            sys.executable,
            str(trace_script),
            "--teacher",
            str(Path(args.weights).resolve()),
            "--student",
            str(Path(final_ckpt).resolve()),
            "--images",
            str(Path(args.target_images).resolve()),
            "--gt-masks",
            str(Path(args.post_trace_gt_masks).resolve()),
            "--out-dir",
            str(trace_out.resolve()),
            "--imgsz",
            str(args.imgsz),
            "--device",
            str(args.device),
            "--student-conf",
            "0.25",
            "--tau-o2o",
            str(args.tau_o2o),
            "--tau-o2m",
            str(args.tau_o2m),
            "--tau-no",
            str(args.tau_no),
            "--tau-dup",
            str(args.tau_dup),
            "--bdl-tau-match",
            str(args.bdl_tau_match),
            "--bdl-max-witnesses",
            str(args.bdl_max_witnesses),
            "--mask-thr",
            str(args.mask_thr),
            "--stability-low",
            str(args.stability_low),
            "--stability-high",
            str(args.stability_high),
            "--mask-rel-thr",
            str(args.mask_rel_thr),
            "--min-mask-pixels",
            str(args.min_mask_pixels),
            "--bdl-boundary-kernel",
            str(args.bdl_boundary_kernel),
        ]
        if args.post_trace_max_images > 0:
            cmd += [
                "--max-images",
                str(args.post_trace_max_images),
            ]

        print()
        print("[POST-TRACE] Training is frozen. Running evaluation-only Teacher/Student visualization...")
        subprocess.run(cmd, check=True)
        print("[POST-TRACE] Saved:", trace_out)

    print()
    print(
        "[PASS] Dense SFSeg run completed"
    )


if __name__ == "__main__":
    main()
