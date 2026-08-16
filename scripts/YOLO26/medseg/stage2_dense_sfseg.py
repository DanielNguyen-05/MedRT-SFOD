#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
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


# ================================================================
# Main
# ================================================================

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
        "Mask-DHF rel  :",
        args.mask_rel_thr,
    )

    print(
        "stability     :",
        args.stability_low,
        args.stability_high,
    )

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
        "var/cov obj.   : unchanged"
    )

    global_step = 0

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
                "fg_tokens": 0.0,
                "hard_bg_tokens": 0.0,
                "easy_bg_tokens": 0.0,
                "core_fallbacks": 0.0,
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

                (
                    labels,
                    masks,
                    ps,
                ) = (
                    generate_mask_dhf_pseudo_masks(
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
                    )
                )

                valid = [
                    i
                    for i, x
                    in enumerate(labels)
                    if x.shape[0] > 0
                ]

                if not valid:
                    skipped += 1
                    global_step += 1
                    continue

                strong_valid = strong[
                    valid
                ]

                labels_valid = [
                    labels[i]
                    for i in valid
                ]

                masks_valid = [
                    masks[i]
                    for i in valid
                ]

                avg_conf = (
                    average_confidence(
                        labels_valid
                    )
                )

                hook.latest = None

                student_outputs = student(
                    strong_valid
                )

                feats = hook.latest

                if feats is None:
                    raise RuntimeError(
                        "MARD feature hook "
                        "captured no features"
                    )

                pseudo_batch = (
                    build_pseudo_batch(
                        labels_valid,
                        masks_valid,
                        strong_valid.shape,
                    )
                )

                det_vec, loss_items = (
                    criterion(
                        student_outputs,
                        pseudo_batch,
                    )
                )

                sfseg_loss = (
                    det_vec.sum()
                )

                if args.mard_mode == "box":
                    mard_loss, mard_stats = (
                        compute_mard_loss(
                            feats,
                            labels_valid,
                            int(
                                strong_valid
                                .shape[2]
                            ),
                            int(
                                strong_valid
                                .shape[3]
                            ),
                            args,
                        )
                    )
                else:
                    mard_loss, mard_stats = (
                        compute_segmard_loss(
                            feats=feats,
                            pseudo_labels=labels_valid,
                            pseudo_masks=masks_valid,
                            h_pad=int(
                                strong_valid
                                .shape[2]
                            ),
                            w_pad=int(
                                strong_valid
                                .shape[3]
                            ),
                            args=args,
                        )
                    )

                lambda_mard = (
                    mard_weight(
                        args,
                        global_step,
                        len(loader),
                        avg_conf,
                    )
                )

                total_loss = (
                    sfseg_loss
                    + lambda_mard
                    * mard_loss
                )

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
                                f"FG/HBG/EBG="
                                f"{int(mard_stats.get('fg_tokens', 0))}/"
                                f"{int(mard_stats.get('hard_bg_tokens', 0))}/"
                                f"{int(mard_stats.get('easy_bg_tokens', 0))} "
                                f"fallback="
                                f"{int(mard_stats.get('core_fallbacks', 0))} "
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
                f"loss={totals['loss']/denom:.4f} "
                f"seg={totals['seg_loss']/denom:.4f} "
                f"semseg={totals['semseg_loss']/denom:.4f} "
                f"mard={totals['mard']/denom:.4f} "
                f"lambda_avg={totals['lambda']/denom:.6f}"
                + (
                    (
                        f" FG/HBG/EBG="
                        f"{totals['fg_tokens']/denom:.1f}/"
                        f"{totals['hard_bg_tokens']/denom:.1f}/"
                        f"{totals['easy_bg_tokens']/denom:.1f}"
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

    print()
    print(
        "[PASS] Dense SFSeg run completed"
    )


if __name__ == "__main__":
    main()
