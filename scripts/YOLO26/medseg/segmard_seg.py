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


def weighted_variance_loss(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    gamma: float,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Weighted counterpart of variance_loss; identical when all weights are one."""
    if tokens.numel() == 0 or tokens.shape[0] < 2:
        return tokens.new_zeros(())
    w = weights.to(device=tokens.device, dtype=tokens.dtype).flatten().clamp_min(0.0)
    if w.numel() != tokens.shape[0]:
        raise ValueError("Token/weight length mismatch")
    sw = w.sum()
    if float(sw.detach().item()) <= EPS:
        return tokens.new_zeros(())
    mu = (tokens * w[:, None]).sum(dim=0) / sw
    var = ((tokens - mu).pow(2) * w[:, None]).sum(dim=0) / sw
    std = torch.sqrt(var + eps)
    return torch.relu(gamma - std).mean()


def weighted_covariance_loss(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Reliability-weighted standardized covariance.

    The weighted degrees-of-freedom denominator reduces exactly to n-1 when
    all weights are one, matching covariance_loss.
    """
    if tokens.numel() == 0 or tokens.shape[0] < 2:
        return tokens.new_zeros(())
    n, c = tokens.shape
    w = weights.to(device=tokens.device, dtype=tokens.dtype).flatten().clamp_min(0.0)
    if w.numel() != n:
        raise ValueError("Token/weight length mismatch")
    sw = w.sum()
    if float(sw.detach().item()) <= EPS:
        return tokens.new_zeros(())

    mu = (tokens * w[:, None]).sum(dim=0) / sw
    centered = tokens - mu
    var = (centered.pow(2) * w[:, None]).sum(dim=0) / sw
    z = centered / (torch.sqrt(var.clamp_min(0.0))[None, :] + eps)

    dof = sw - w.pow(2).sum() / sw.clamp_min(EPS)
    cov = (z.T @ (z * w[:, None])) / dof.clamp_min(1.0)
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


def _as_reliability_tensor(values, device: torch.device) -> torch.Tensor:
    out = torch.as_tensor(values, device=device, dtype=torch.float32).flatten()
    return out.clamp(0.0, 1.0)


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
    reliabilities: Optional[torch.Tensor] = None,
    reliability_weighting: bool = False,
    return_weights: bool = False,
):
    """
    SegMARD region sampler.

    V1/V2 sampling semantics are preserved exactly:
      - FG comes from the pseudo mask;
      - HBG is the union of in-box / out-of-mask pixels at the assigned level;
      - EBG is outside boxes assigned to the same pyramid level;
      - the original BG token budget is preserved at every level.

    V3 does not change sampling. It only attaches reliability weights:
      - FG token from instance i -> r_i;
      - HBG token -> max r_i among hard-BG regions covering that pixel;
      - EBG token -> 1.
    """
    if not (0.0 <= hard_bg_ratio <= 1.0):
        raise ValueError("hard_bg_ratio must be in [0,1]")

    device = fmap.device
    _, h_f, w_f = fmap.shape
    valid = torch.ones((h_f, w_f), dtype=torch.bool, device=device)

    if reliability_weighting:
        if reliabilities is None:
            raise ValueError("V3 reliability weighting requires per-instance reliabilities")
        reliabilities = _as_reliability_tensor(reliabilities, device)
        if reliabilities.numel() != boxes.shape[0]:
            raise ValueError(
                f"boxes={boxes.shape[0]} reliabilities={reliabilities.numel()} mismatch"
            )
    else:
        reliabilities = torch.ones((boxes.shape[0],), device=device, dtype=torch.float32)

    level_ids = torch.where(levels == target_level)[0]

    # Preserve original per-level MARD background semantics.
    level_box_union = torch.zeros((h_f, w_f), dtype=torch.bool, device=device)
    for idx in level_ids.tolist():
        rect = _feature_rect(boxes[idx], h_f, w_f, h_pad, w_pad)
        if rect is None:
            continue
        x1f, y1f, x2f, y2f = rect
        level_box_union[y1f : y2f + 1, x1f : x2f + 1] = True

    fg_tokens: list[torch.Tensor] = []
    fg_weights: list[torch.Tensor] = []
    hard_bg_union = torch.zeros((h_f, w_f), dtype=torch.bool, device=device)
    hard_bg_weight_map = torch.zeros((h_f, w_f), dtype=torch.float32, device=device)
    core_fallbacks = 0

    for idx in level_ids.tolist():
        box = boxes[idx]
        mask_f = _resize_mask_presence(masks[idx], h_f, w_f)

        rect = _feature_rect(box, h_f, w_f, h_pad, w_pad)
        if rect is None:
            continue
        x1f, y1f, x2f, y2f = rect
        box_mask = torch.zeros_like(mask_f)
        box_mask[y1f : y2f + 1, x1f : x2f + 1] = True
        mask_f = mask_f & box_mask

        core = _binary_erode(mask_f, erode_kernel)
        if not core.any():
            # Never fall back to the full box.
            core = mask_f
            core_fallbacks += 1

        fg_coords = _sample_coords(core, fg_points)
        fg = _tokens_from_coords(fmap, fg_coords)
        if fg is not None:
            fg_tokens.append(fg)
            r = reliabilities[idx].to(dtype=fmap.dtype)
            fg_weights.append(torch.full(
                (fg.shape[0],),
                float(r.detach().item()),
                device=device,
                dtype=fmap.dtype,
            ))

        dilated = _binary_dilate(mask_f, dilate_kernel)
        hard_region = box_mask & (~dilated)
        hard_bg_union |= hard_region

        if reliability_weighting and hard_region.any():
            r_map = hard_region.float() * reliabilities[idx]
            hard_bg_weight_map = torch.maximum(hard_bg_weight_map, r_map)

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
            fallback_region = valid
            fallback_to_easy = True

        extra = _sample_coords(fallback_region, missing)
        if extra is not None:
            if fallback_to_easy:
                easy_coords = extra if easy_coords is None else torch.cat([easy_coords, extra], dim=0)
            else:
                hard_coords = extra if hard_coords is None else torch.cat([hard_coords, extra], dim=0)

    chunks = list(fg_tokens)
    weight_chunks = list(fg_weights)

    hard_tok = _tokens_from_coords(fmap, hard_coords)
    easy_tok = _tokens_from_coords(fmap, easy_coords)

    if hard_tok is not None:
        chunks.append(hard_tok)
        if reliability_weighting:
            hard_w = hard_bg_weight_map[hard_coords[:, 0], hard_coords[:, 1]]
            # A hard-region pixel should have at least one owner. Clamp only for
            # numerical safety; do not invent a confidence floor.
            hard_w = hard_w.to(dtype=fmap.dtype).clamp(0.0, 1.0)
        else:
            hard_w = torch.ones((hard_tok.shape[0],), device=device, dtype=fmap.dtype)
        weight_chunks.append(hard_w)

    if easy_tok is not None:
        chunks.append(easy_tok)
        weight_chunks.append(
            torch.ones((easy_tok.shape[0],), device=device, dtype=fmap.dtype)
        )

    if not chunks:
        stats = {
            "fg": 0,
            "hard_bg": 0,
            "easy_bg": 0,
            "core_fallbacks": int(core_fallbacks),
        }
        if return_weights:
            return None, None, stats
        return None, stats

    tokens = torch.cat(chunks, dim=0)
    token_weights = torch.cat(weight_chunks, dim=0)

    if token_weights.shape[0] != tokens.shape[0]:
        raise RuntimeError("SegMARD token/weight alignment failure")

    stats = {
        "fg": int(sum(x.shape[0] for x in fg_tokens)),
        "hard_bg": 0 if hard_tok is None else int(hard_tok.shape[0]),
        "easy_bg": 0 if easy_tok is None else int(easy_tok.shape[0]),
        "core_fallbacks": int(core_fallbacks),
    }

    if return_weights:
        return tokens, token_weights, stats
    return tokens, stats

def compute_segmard_loss(
    feats: list[torch.Tensor],
    pseudo_labels: list[torch.Tensor],
    pseudo_masks: list[torch.Tensor],
    h_pad: int,
    w_pad: int,
    args,
    pseudo_reliabilities: Optional[list[torch.Tensor]] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Segmentation-guided MARD with optional V3 reliability-weighted statistics."""
    if len(feats) < 3:
        raise ValueError(f"Expected P3/P4/P5 features, got {len(feats)}")
    if len(pseudo_labels) != len(pseudo_masks):
        raise ValueError("pseudo_labels and pseudo_masks batch lengths differ")

    use_rel = bool(getattr(args, "segmard_reliability_weighting", False))
    if use_rel:
        if pseudo_reliabilities is None:
            raise ValueError("V3 requires pseudo_reliabilities")
        if len(pseudo_reliabilities) != len(pseudo_labels):
            raise ValueError("pseudo_reliabilities batch length mismatch")

    total = feats[0].new_zeros(())
    stats: dict[str, float] = {}
    stride3 = float(w_pad) / float(feats[0].shape[3])
    stride4 = float(w_pad) / float(feats[1].shape[3])

    global_fg = 0
    global_hbg = 0
    global_ebg = 0
    global_fallback = 0
    global_weight_sum = 0.0
    global_weight_count = 0

    for level_idx, fmap in enumerate(feats[:3]):
        tokens_all: list[torch.Tensor] = []
        weights_all: list[torch.Tensor] = []
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

            rel = None
            if use_rel:
                rel = _as_reliability_tensor(pseudo_reliabilities[b], fmap.device)
                if rel.shape[0] != labels.shape[0]:
                    raise RuntimeError(
                        f"Image {b}: labels={labels.shape[0]} reliabilities={rel.shape[0]} mismatch"
                    )

            boxes = labels[:, :4].to(fmap.device)
            confs = labels[:, 4].to(fmap.device)
            keep = confs >= float(args.mard_box_conf)
            if keep.sum() == 0:
                continue

            keep_idx = torch.where(keep)[0]
            if keep_idx.numel() > int(args.mard_topk_boxes):
                order = torch.argsort(
                    confs[keep_idx],
                    descending=True,
                )[: int(args.mard_topk_boxes)]
                keep_idx = keep_idx[order]

            boxes = boxes[keep_idx]
            masks = masks[keep_idx]
            if use_rel:
                rel = rel[keep_idx]

            levels = assign_boxes_to_levels(
                boxes,
                stride3=stride3,
                stride4=stride4,
                eta=float(args.mard_eta),
            )

            if use_rel:
                tokens, token_weights, s = sample_segmard_level_tokens_for_image(
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
                    reliabilities=rel,
                    reliability_weighting=True,
                    return_weights=True,
                )
                if tokens is not None:
                    tokens_all.append(tokens)
                    weights_all.append(token_weights)
            else:
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
            if use_rel:
                w = torch.cat(weights_all, dim=0)
                if w.shape[0] != z.shape[0]:
                    raise RuntimeError("V3 level token/weight alignment failure")
                var = weighted_variance_loss(
                    z, w, gamma=float(args.mard_gamma)
                )
                cov = weighted_covariance_loss(z, w)
                level_weight_mean = float(w.detach().mean().item())
                global_weight_sum += float(w.detach().sum().item())
                global_weight_count += int(w.numel())
            else:
                var = variance_loss(z, gamma=float(args.mard_gamma))
                cov = covariance_loss(z)
                level_weight_mean = 1.0
        else:
            var = fmap.new_zeros(())
            cov = fmap.new_zeros(())
            level_weight_mean = 0.0

        level_loss = float(args.mard_alpha) * var + float(args.mard_beta) * cov
        total = total + level_loss

        p = f"p{level_idx + 3}"
        stats[f"{p}_var"] = float(var.detach().item())
        stats[f"{p}_cov"] = float(cov.detach().item())
        stats[f"{p}_fg_tokens"] = float(level_fg)
        stats[f"{p}_hard_bg_tokens"] = float(level_hbg)
        stats[f"{p}_easy_bg_tokens"] = float(level_ebg)
        stats[f"{p}_core_fallbacks"] = float(level_fallback)
        stats[f"{p}_weight_mean"] = float(level_weight_mean)

        global_fg += level_fg
        global_hbg += level_hbg
        global_ebg += level_ebg
        global_fallback += level_fallback

    stats["segmard"] = float(total.detach().item())
    stats["fg_tokens"] = float(global_fg)
    stats["hard_bg_tokens"] = float(global_hbg)
    stats["easy_bg_tokens"] = float(global_ebg)
    stats["core_fallbacks"] = float(global_fallback)
    stats["reliability_weighting"] = float(use_rel)
    stats["token_weight_mean"] = (
        global_weight_sum / global_weight_count
        if global_weight_count > 0
        else (1.0 if not use_rel else 0.0)
    )
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
    g.add_argument(
        "--segmard-reliability-weighting",
        action="store_true",
        help=(
            "V3: keep SegMARD sampling unchanged but weight FG/HBG representation "
            "statistics by per-instance mask reliability; EBG weight remains 1."
        ),
    )
