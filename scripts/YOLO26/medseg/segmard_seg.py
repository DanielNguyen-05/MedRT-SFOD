#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


EPS = 1e-6


def variance_loss(tokens: torch.Tensor, gamma: float, eps: float = 1e-4) -> torch.Tensor:
    if tokens.numel() == 0 or tokens.shape[0] < 2:
        return tokens.new_zeros(())
    std = torch.sqrt(tokens.var(dim=0, unbiased=False) + eps)
    return torch.relu(gamma - std).mean()


def covariance_loss(tokens: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    if tokens.numel() == 0 or tokens.shape[0] < 2:
        return tokens.new_zeros(())
    n, c = tokens.shape
    z = tokens - tokens.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, unbiased=False, keepdim=True) + eps)
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return off_diag.pow(2).sum() / (c * (c - 1) + EPS)


def assign_boxes_to_levels(
    boxes: torch.Tensor,
    stride3: float,
    stride4: float,
    eta: float,
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    sizes = torch.sqrt(
        (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
        * (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
    )
    levels = torch.empty_like(sizes, dtype=torch.long)
    levels[sizes <= eta * stride3] = 0
    levels[(sizes > eta * stride3) & (sizes <= eta * stride4)] = 1
    levels[sizes > eta * stride4] = 2
    return levels


def _feature_rect(
    box: torch.Tensor,
    h_f: int,
    w_f: int,
    h_pad: int,
    w_pad: int,
) -> Optional[tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box.float()
    x1f = int(torch.floor(x1 * w_f / max(w_pad, 1)).item())
    y1f = int(torch.floor(y1 * h_f / max(h_pad, 1)).item())
    x2f = int(torch.ceil(x2 * w_f / max(w_pad, 1)).item()) - 1
    y2f = int(torch.ceil(y2 * h_f / max(h_pad, 1)).item()) - 1

    x1f = max(0, min(x1f, w_f - 1))
    x2f = max(0, min(x2f, w_f - 1))
    y1f = max(0, min(y1f, h_f - 1))
    y2f = max(0, min(y2f, h_f - 1))
    if x2f < x1f or y2f < y1f:
        return None
    return x1f, y1f, x2f, y2f


def _as_mask_tensor(masks, device: torch.device) -> torch.Tensor:
    """Normalize pseudo masks to [N,H,W] bool on the feature device."""
    if isinstance(masks, torch.Tensor):
        out = masks
    elif isinstance(masks, (list, tuple)):
        if not masks:
            return torch.zeros((0, 1, 1), dtype=torch.bool, device=device)
        out = torch.stack([torch.as_tensor(x) for x in masks], dim=0)
    else:
        out = torch.as_tensor(masks)

    if out.ndim == 2:
        out = out.unsqueeze(0)
    if out.ndim == 4 and out.shape[1] == 1:
        out = out[:, 0]
    if out.ndim != 3:
        raise ValueError(f"Expected pseudo masks [N,H,W], got shape={tuple(out.shape)}")
    return out.to(device=device).bool()


def _resize_mask_presence(mask: torch.Tensor, h_f: int, w_f: int) -> torch.Tensor:
    """Project an input-space binary mask to a feature map while preserving tiny objects."""
    x = mask.float()[None, None]
    h, w = mask.shape[-2:]
    if h >= h_f and w >= w_f:
        y = F.adaptive_max_pool2d(x, output_size=(h_f, w_f))
    else:
        y = F.interpolate(x, size=(h_f, w_f), mode="nearest")
    return y[0, 0] > 0.5


def _binary_dilate(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return mask
    if kernel % 2 == 0:
        raise ValueError("SegMARD morphology kernel must be odd")
    x = mask.float()[None, None]
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return y[0, 0] > 0.5


def _binary_erode(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return mask
    if kernel % 2 == 0:
        raise ValueError("SegMARD morphology kernel must be odd")
    x = (~mask).float()[None, None]
    y = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return ~(y[0, 0] > 0.5)


def _sample_coords(region: torch.Tensor, n: int) -> Optional[torch.Tensor]:
    if n <= 0:
        return None
    coords = region.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    # Match original MARD semantics: random sampling with replacement.
    idx = torch.randint(0, coords.shape[0], (n,), device=region.device)
    return coords[idx]


def _tokens_from_coords(fmap: torch.Tensor, coords: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if coords is None or coords.numel() == 0:
        return None
    return fmap[:, coords[:, 0], coords[:, 1]].T


def sample_segmard_level_tokens_for_image(
    fmap: torch.Tensor,
    boxes: torch.Tensor,
    masks: torch.Tensor,
    levels: torch.Tensor,
    target_level: int,
    h_pad: int,
    w_pad: int,
    fg_points: int,
    bg_points: int,
    erode_kernel: int = 1,
    dilate_kernel: int = 1,
    hard_bg_ratio: float = 0.0,
) -> tuple[Optional[torch.Tensor], dict[str, int]]:
    """
    SegMARD region sampler.

    Default (kernel=1, hard_bg_ratio=0) is the clean first ablation:
      - foreground is sampled from pseudo segmentation masks instead of box interiors;
      - per-level background sampling is kept identical to original MARD:
        outside the pseudo boxes assigned to that same pyramid level;
      - levels with no assigned object still contribute the original BG token budget.

    With erode/dilate >1 and hard_bg_ratio>0 it becomes region-aware:
      - reliable FG core = eroded pseudo mask;
      - hard BG = inside box but outside dilated pseudo mask;
      - easy BG = outside all selected pseudo boxes.
    """
    if not (0.0 <= hard_bg_ratio <= 1.0):
        raise ValueError("hard_bg_ratio must be in [0,1]")

    device = fmap.device
    _, h_f, w_f = fmap.shape
    valid = torch.ones((h_f, w_f), dtype=torch.bool, device=device)

    level_ids = torch.where(levels == target_level)[0]

    # IMPORTANT FOR CLEAN ABLATION:
    # Match the original MARD background semantics exactly.
    # At each pyramid level, background is defined outside only the pseudo boxes
    # assigned to that same level. Even when no object is assigned to a level,
    # original MARD still samples the full-map background budget at that level.
    level_box_union = torch.zeros((h_f, w_f), dtype=torch.bool, device=device)
    for idx in level_ids.tolist():
        rect = _feature_rect(boxes[idx], h_f, w_f, h_pad, w_pad)
        if rect is None:
            continue
        x1f, y1f, x2f, y2f = rect
        level_box_union[y1f : y2f + 1, x1f : x2f + 1] = True

    fg_tokens: list[torch.Tensor] = []
    hard_bg_union = torch.zeros((h_f, w_f), dtype=torch.bool, device=device)
    core_fallbacks = 0

    for idx in level_ids.tolist():
        box = boxes[idx]
        mask_f = _resize_mask_presence(masks[idx], h_f, w_f)

        # Restrict mask evidence to its teacher box to avoid accidental spillover.
        rect = _feature_rect(box, h_f, w_f, h_pad, w_pad)
        if rect is None:
            continue
        x1f, y1f, x2f, y2f = rect
        box_mask = torch.zeros_like(mask_f)
        box_mask[y1f : y2f + 1, x1f : x2f + 1] = True
        mask_f = mask_f & box_mask

        core = _binary_erode(mask_f, erode_kernel)
        if not core.any():
            # Do not fall back to the whole box. Fall back only to the pseudo mask.
            core = mask_f
            core_fallbacks += 1

        fg_coords = _sample_coords(core, fg_points)
        fg = _tokens_from_coords(fmap, fg_coords)
        if fg is not None:
            fg_tokens.append(fg)

        dilated = _binary_dilate(mask_f, dilate_kernel)
        hard_bg_union |= box_mask & (~dilated)

    easy_bg = valid & (~level_box_union)

    n_hard = int(round(bg_points * hard_bg_ratio))
    n_easy = int(bg_points - n_hard)

    hard_coords = _sample_coords(hard_bg_union, n_hard)
    easy_coords = _sample_coords(easy_bg, n_easy)

    # Keep total BG budget stable if one region is absent.
    hard_have = 0 if hard_coords is None else int(hard_coords.shape[0])
    easy_have = 0 if easy_coords is None else int(easy_coords.shape[0])
    missing = bg_points - hard_have - easy_have
    if missing > 0:
        if easy_bg.any():
            fallback_region = easy_bg
            fallback_to_easy = True
        elif hard_bg_union.any():
            fallback_region = hard_bg_union
            fallback_to_easy = False
        else:
            # Match original MARD fallback when no background coordinates exist:
            # sample from the valid feature map rather than changing the token budget.
            fallback_region = valid
            fallback_to_easy = True

        extra = _sample_coords(fallback_region, missing)
        if extra is not None:
            if fallback_to_easy:
                easy_coords = extra if easy_coords is None else torch.cat([easy_coords, extra], dim=0)
            else:
                hard_coords = extra if hard_coords is None else torch.cat([hard_coords, extra], dim=0)

    chunks = list(fg_tokens)
    hard_tok = _tokens_from_coords(fmap, hard_coords)
    easy_tok = _tokens_from_coords(fmap, easy_coords)
    if hard_tok is not None:
        chunks.append(hard_tok)
    if easy_tok is not None:
        chunks.append(easy_tok)

    if not chunks:
        return None, {"fg": 0, "hard_bg": 0, "easy_bg": 0, "core_fallbacks": core_fallbacks}

    stats = {
        "fg": int(sum(x.shape[0] for x in fg_tokens)),
        "hard_bg": 0 if hard_tok is None else int(hard_tok.shape[0]),
        "easy_bg": 0 if easy_tok is None else int(easy_tok.shape[0]),
        "core_fallbacks": int(core_fallbacks),
    }
    return torch.cat(chunks, dim=0), stats


def compute_segmard_loss(
    feats: list[torch.Tensor],
    pseudo_labels: list[torch.Tensor],
    pseudo_masks: list[torch.Tensor],
    h_pad: int,
    w_pad: int,
    args,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Segmentation-driven replacement for original box-guided MARD sampling."""
    if len(feats) < 3:
        raise ValueError(f"Expected P3/P4/P5 features, got {len(feats)}")
    if len(pseudo_labels) != len(pseudo_masks):
        raise ValueError("pseudo_labels and pseudo_masks batch lengths differ")

    total = feats[0].new_zeros(())
    stats: dict[str, float] = {}
    stride3 = float(w_pad) / float(feats[0].shape[3])
    stride4 = float(w_pad) / float(feats[1].shape[3])

    global_fg = 0
    global_hbg = 0
    global_ebg = 0
    global_fallback = 0

    for level_idx, fmap in enumerate(feats[:3]):
        tokens_all: list[torch.Tensor] = []
        level_fg = level_hbg = level_ebg = level_fallback = 0

        for b in range(fmap.shape[0]):
            labels = pseudo_labels[b]
            if labels.numel() == 0:
                continue

            masks = _as_mask_tensor(pseudo_masks[b], fmap.device)
            if masks.shape[0] != labels.shape[0]:
                raise RuntimeError(
                    f"Image {b}: labels={labels.shape[0]} masks={masks.shape[0]} mismatch"
                )

            boxes = labels[:, :4].to(fmap.device)
            confs = labels[:, 4].to(fmap.device)
            keep = confs >= float(args.mard_box_conf)
            if keep.sum() == 0:
                continue

            keep_idx = torch.where(keep)[0]
            if keep_idx.numel() > int(args.mard_topk_boxes):
                order = torch.argsort(confs[keep_idx], descending=True)[: int(args.mard_topk_boxes)]
                keep_idx = keep_idx[order]

            boxes = boxes[keep_idx]
            masks = masks[keep_idx]

            levels = assign_boxes_to_levels(
                boxes,
                stride3=stride3,
                stride4=stride4,
                eta=float(args.mard_eta),
            )

            tokens, s = sample_segmard_level_tokens_for_image(
                fmap=fmap[b],
                boxes=boxes,
                masks=masks,
                levels=levels,
                target_level=level_idx,
                h_pad=h_pad,
                w_pad=w_pad,
                fg_points=int(args.mard_fg_points),
                bg_points=int(args.mard_bg_points),
                erode_kernel=int(args.segmard_erode_kernel),
                dilate_kernel=int(args.segmard_dilate_kernel),
                hard_bg_ratio=float(args.segmard_hard_bg_ratio),
            )
            if tokens is not None:
                tokens_all.append(tokens)

            level_fg += s["fg"]
            level_hbg += s["hard_bg"]
            level_ebg += s["easy_bg"]
            level_fallback += s["core_fallbacks"]

        if tokens_all:
            z = torch.cat(tokens_all, dim=0)
            var = variance_loss(z, gamma=float(args.mard_gamma))
            cov = covariance_loss(z)
        else:
            var = fmap.new_zeros(())
            cov = fmap.new_zeros(())

        level_loss = float(args.mard_alpha) * var + float(args.mard_beta) * cov
        total = total + level_loss

        p = f"p{level_idx + 3}"
        stats[f"{p}_var"] = float(var.detach().item())
        stats[f"{p}_cov"] = float(cov.detach().item())
        stats[f"{p}_fg_tokens"] = float(level_fg)
        stats[f"{p}_hard_bg_tokens"] = float(level_hbg)
        stats[f"{p}_easy_bg_tokens"] = float(level_ebg)
        stats[f"{p}_core_fallbacks"] = float(level_fallback)

        global_fg += level_fg
        global_hbg += level_hbg
        global_ebg += level_ebg
        global_fallback += level_fallback

    stats["segmard"] = float(total.detach().item())
    stats["fg_tokens"] = float(global_fg)
    stats["hard_bg_tokens"] = float(global_hbg)
    stats["easy_bg_tokens"] = float(global_ebg)
    stats["core_fallbacks"] = float(global_fallback)
    return total, stats


def add_segmard_args(parser) -> None:
    g = parser.add_argument_group("SegMARD mask-guided sampling")
    g.add_argument(
        "--segmard-erode-kernel",
        type=int,
        default=1,
        help="Odd morphology kernel. 1 means foreground = pseudo mask exactly.",
    )
    g.add_argument(
        "--segmard-dilate-kernel",
        type=int,
        default=1,
        help="Odd morphology kernel used to define mask-adjacent hard background.",
    )
    g.add_argument(
        "--segmard-hard-bg-ratio",
        type=float,
        default=0.0,
        help="Fraction of original MARD BG budget sampled inside box but outside dilated pseudo mask.",
    )