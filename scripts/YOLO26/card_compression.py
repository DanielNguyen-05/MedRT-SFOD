"""Reliability-guided compression controller for RT-SFOD/MedRT-SFOD.

This revision fixes two important problems in the earlier ``card_compression.py``:

1) A pruned student must still have *dense latent parameters* if the teacher is
   updated by parameter-wise EMA.  Therefore pruning is represented as a
   forward-time feature-channel gate; it NEVER zeros the stored student
   parameters in-place.  The teacher consequently receives dense FP weights
   through EMA, exactly matching the intended asymmetric teacher/student story.

2) Zeroing only Conv2d filters does not truly zero a YOLO feature channel when a
   following BatchNorm has affine/running-statistics.  This implementation gates
   the output of Conv-BN-activation wrapper modules, i.e. after BN/activation,
   which is a faithful simulation of removing that output feature channel.

Weight fake quantization is intentionally described as *weight fake-QAT* rather
than deployable INT8.  Actual latency/size claims require a later graph-aware
physical pruning + integer export step; this module does not pretend otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Schedule
# -----------------------------------------------------------------------------


def reliability_ramp(
    target: float,
    optimizer_step: int,
    steps_per_epoch: int,
    warmup_epochs: float,
    reliability: float,
    gate_threshold: float,
    delay_epochs: float = 0.0,
) -> float:
    """Compression strength in ``[0, target]``.

    Compression is delayed explicitly, then linearly ramped and finally gated
    by pseudo-label reliability.  ``optimizer_step`` should count only steps
    that actually update the student, rather than skipped target batches.
    """
    if target <= 0.0:
        return 0.0
    steps_per_epoch = max(int(steps_per_epoch), 1)
    delay_steps = max(0, int(round(delay_epochs * steps_per_epoch)))
    if optimizer_step < delay_steps:
        return 0.0

    local_step = optimizer_step - delay_steps
    warmup_steps = max(1, int(round(warmup_epochs * steps_per_epoch)))
    ramp = min(1.0, float(local_step) / float(warmup_steps))
    gate = (float(reliability) - gate_threshold) / max(1.0 - gate_threshold, 1e-6)
    gate = float(np.clip(gate, 0.0, 1.0))
    return float(target) * ramp * gate


# -----------------------------------------------------------------------------
# STE fake quantization
# -----------------------------------------------------------------------------


class _FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, num_bits: int) -> torch.Tensor:
        if num_bits < 2:
            raise ValueError("num_bits must be >= 2")
        qmax = 2 ** (num_bits - 1) - 1
        max_val = x.detach().abs().max().clamp(min=1e-8)
        scale = max_val / qmax
        return torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def fake_quantize_weight(weight: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    return _FakeQuantSTE.apply(weight, int(num_bits))


@dataclass
class _SharedQuantState:
    ratio: float = 0.0
    num_bits: int = 8


def _quantized_conv_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    state: _SharedQuantState = self._card_v2_quant_state
    ratio = float(state.ratio)
    if ratio <= 0.0:
        weight = self.weight
    else:
        q_weight = fake_quantize_weight(self.weight, state.num_bits)
        weight = self.weight + ratio * (q_weight - self.weight)
    return self._conv_forward(x, weight, self.bias)


def _gated_wrapper_forward(self: nn.Module, x: torch.Tensor):
    y = self._card_v2_original_forward(x)
    mask = self._card_v2_channel_mask
    if isinstance(y, torch.Tensor) and y.dim() >= 2 and y.shape[1] == mask.numel():
        shape = [1, mask.numel()] + [1] * (y.dim() - 2)
        y = y * mask.to(device=y.device, dtype=y.dtype).view(*shape)
    return y


# -----------------------------------------------------------------------------
# Structured feature-channel gating
# -----------------------------------------------------------------------------


DEFAULT_EXCLUDE = (
    r"^model\.0(?:\.|$)",  # stem
    r"\.dfl(?:\.|$)",
    r"one2one",
    r"one2many",
)


@dataclass
class GatedLayer:
    name: str
    wrapper: nn.Module
    conv: nn.Conv2d
    mask: torch.Tensor


class FeatureChannelPruner:
    """Structured output-channel gating while preserving dense latent weights.

    Eligible modules are Conv-BN-like wrappers exposing ``.conv`` (Conv2d) and
    ``.bn`` (BatchNorm2d).  The gate is applied *after* the wrapper's normal
    forward so a pruned channel is exactly zero even when BN has non-zero beta.
    """

    def __init__(
        self,
        model: nn.Module,
        exclude_patterns: Iterable[str] = DEFAULT_EXCLUDE,
        min_channels: int = 8,
        allow_regrowth: bool = False,
    ) -> None:
        self.model = model
        self.patterns = [re.compile(p) for p in exclude_patterns]
        self.min_channels = int(min_channels)
        self.allow_regrowth = bool(allow_regrowth)
        self.layers: dict[str, GatedLayer] = {}
        self._installed = False
        self._discover()

    def _head_prefixes(self) -> list[str]:
        prefixes: list[str] = []
        top = getattr(self.model, "model", None)
        if top is not None and len(top):
            head = top[-1]
            for name, module in self.model.named_modules():
                if module is head:
                    prefixes.append(name)
                    break
        return prefixes

    def _excluded(self, name: str, head_prefixes: list[str]) -> bool:
        if any(p.search(name) for p in self.patterns):
            return True
        return any(name == p or name.startswith(p + ".") for p in head_prefixes if p)

    def _discover(self) -> None:
        head_prefixes = self._head_prefixes()
        for name, module in self.model.named_modules():
            if self._excluded(name, head_prefixes):
                continue
            conv = getattr(module, "conv", None)
            bn = getattr(module, "bn", None)
            if not isinstance(conv, nn.Conv2d) or not isinstance(bn, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                continue
            if conv.out_channels < self.min_channels:
                continue
            # Depthwise channels require coupled graph surgery at export; skip
            # them in the first publishable version so the sparsity definition
            # stays interpretable.
            if conv.groups == conv.in_channels == conv.out_channels:
                continue
            mask = torch.ones(conv.out_channels, dtype=torch.bool, device=conv.weight.device)
            self.layers[name] = GatedLayer(name=name, wrapper=module, conv=conv, mask=mask)

    def install(self) -> None:
        if self._installed:
            return
        for layer in self.layers.values():
            wrapper = layer.wrapper
            if hasattr(wrapper, "_card_v2_original_forward"):
                continue
            wrapper._card_v2_original_forward = wrapper.forward
            wrapper._card_v2_channel_mask = layer.mask
            wrapper.forward = _gated_wrapper_forward.__get__(wrapper, wrapper.__class__)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for layer in self.layers.values():
            wrapper = layer.wrapper
            original = getattr(wrapper, "_card_v2_original_forward", None)
            if original is not None:
                wrapper.forward = original
            for attr in ("_card_v2_original_forward", "_card_v2_channel_mask"):
                if hasattr(wrapper, attr):
                    delattr(wrapper, attr)
        self._installed = False

    @torch.no_grad()
    def update_masks(self, target_sparsity: float) -> float:
        target_sparsity = float(np.clip(target_sparsity, 0.0, 0.999))
        total = 0
        pruned = 0
        for name, layer in self.layers.items():
            importance = layer.conv.weight.detach().abs().sum(dim=(1, 2, 3))
            k = min(int(round(target_sparsity * importance.numel())), importance.numel() - 1)
            candidate = torch.ones_like(importance, dtype=torch.bool)
            if k > 0:
                candidate[torch.topk(importance, k, largest=False).indices] = False

            old = layer.mask.to(candidate.device)
            new = candidate if self.allow_regrowth else (old & candidate)
            layer.mask = new
            if self._installed:
                layer.wrapper._card_v2_channel_mask = new
            total += new.numel()
            pruned += int((~new).sum().item())
        return pruned / max(total, 1)

    def sparsity(self) -> float:
        total = sum(layer.mask.numel() for layer in self.layers.values())
        pruned = sum(int((~layer.mask).sum().item()) for layer in self.layers.values())
        return pruned / max(total, 1)

    def state_dict(self) -> dict:
        return {
            "allow_regrowth": self.allow_regrowth,
            "masks": {name: layer.mask.detach().cpu() for name, layer in self.layers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        masks = state.get("masks", {})
        for name, mask in masks.items():
            if name not in self.layers:
                continue
            dst = self.layers[name]
            mask = mask.to(device=dst.conv.weight.device, dtype=torch.bool)
            if mask.numel() != dst.conv.out_channels:
                raise ValueError(f"Mask shape mismatch for {name}: {mask.numel()} vs {dst.conv.out_channels}")
            dst.mask = mask
            if self._installed:
                dst.wrapper._card_v2_channel_mask = mask


# -----------------------------------------------------------------------------
# Weight fake-QAT patching
# -----------------------------------------------------------------------------


class WeightFakeQuantizer:
    def __init__(self, model: nn.Module, num_bits: int = 8, exclude_patterns: Iterable[str] = DEFAULT_EXCLUDE):
        self.model = model
        self.state = _SharedQuantState(ratio=0.0, num_bits=int(num_bits))
        self.patterns = [re.compile(p) for p in exclude_patterns]
        self.convs: dict[str, nn.Conv2d] = {}
        self._installed = False
        self._discover()

    def _discover(self) -> None:
        top = getattr(self.model, "model", None)
        head = top[-1] if top is not None and len(top) else None
        head_prefix = None
        if head is not None:
            for name, module in self.model.named_modules():
                if module is head:
                    head_prefix = name
                    break
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Conv2d):
                continue
            if any(p.search(name) for p in self.patterns):
                continue
            if head_prefix and (name == head_prefix or name.startswith(head_prefix + ".")):
                continue
            self.convs[name] = module

    def install(self) -> None:
        if self._installed:
            return
        for conv in self.convs.values():
            if hasattr(conv, "_card_v2_original_forward"):
                continue
            conv._card_v2_original_forward = conv.forward
            conv._card_v2_quant_state = self.state
            conv.forward = _quantized_conv_forward.__get__(conv, nn.Conv2d)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for conv in self.convs.values():
            original = getattr(conv, "_card_v2_original_forward", None)
            if original is not None:
                conv.forward = original
            for attr in ("_card_v2_original_forward", "_card_v2_quant_state"):
                if hasattr(conv, attr):
                    delattr(conv, attr)
        self._installed = False

    def set_ratio(self, ratio: float) -> None:
        self.state.ratio = float(np.clip(ratio, 0.0, 1.0))


# -----------------------------------------------------------------------------
# Controller
# -----------------------------------------------------------------------------


class CardControllerV2:
    """Reliability-gated student compression with dense latent parameters.

    Call ``prepare_step`` BEFORE the student forward so the current target
    reliability controls the current forward.  ``after_optimizer_step`` is
    intentionally a no-op for pruning: no parameter needs to be re-zeroed.
    """

    def __init__(self, student_model: nn.Module, args: argparse.Namespace):
        self.model = student_model
        self.args = args
        self.enabled = bool(getattr(args, "card_enable", False))
        self.pruner: Optional[FeatureChannelPruner] = None
        self.quantizer: Optional[WeightFakeQuantizer] = None
        self.current_stats: dict[str, float] = {}
        self._last_mask_update_step = -1
        if self.enabled:
            self.pruner = FeatureChannelPruner(
                student_model,
                allow_regrowth=bool(getattr(args, "card_allow_regrowth", False)),
            )
            self.quantizer = WeightFakeQuantizer(student_model, num_bits=args.card_quant_bits)
            self.install()

    def install(self) -> None:
        if not self.enabled:
            return
        self.quantizer.install()
        self.pruner.install()

    def uninstall(self) -> None:
        if not self.enabled:
            return
        self.pruner.uninstall()
        self.quantizer.uninstall()

    @contextlib.contextmanager
    def suspended_for_serialization(self) -> Iterator[None]:
        """Temporarily restore vanilla forwards while a full model is pickled.

        The accompanying CARD state MUST also be saved, because the dense
        checkpoint alone intentionally contains latent (unmasked) weights.
        """
        if not self.enabled:
            yield
            return
        self.uninstall()
        try:
            yield
        finally:
            self.install()

    def prepare_step(self, optimizer_step: int, steps_per_epoch: int, reliability: float) -> dict[str, float]:
        if not self.enabled:
            self.current_stats = {}
            return self.current_stats

        prune_target = reliability_ramp(
            self.args.card_prune_target,
            optimizer_step,
            steps_per_epoch,
            self.args.card_warmup_epochs,
            reliability,
            self.args.card_gate_threshold,
            self.args.card_delay_epochs,
        )
        quant_ratio = reliability_ramp(
            self.args.card_quant_target_ratio,
            optimizer_step,
            steps_per_epoch,
            self.args.card_warmup_epochs,
            reliability,
            self.args.card_gate_threshold,
            self.args.card_delay_epochs,
        )
        self.quantizer.set_ratio(quant_ratio)

        interval = max(int(self.args.card_prune_interval), 1)
        if optimizer_step == 0 or optimizer_step - self._last_mask_update_step >= interval:
            self.pruner.update_masks(prune_target)
            self._last_mask_update_step = optimizer_step

        self.current_stats = {
            "card_prune_target": float(prune_target),
            "card_sparsity": float(self.pruner.sparsity()),
            "card_quant_ratio": float(quant_ratio),
            "card_reliability": float(reliability),
        }
        return dict(self.current_stats)

    def after_optimizer_step(self) -> None:
        # Dense latent parameters are never zeroed.  This is the key property
        # that keeps teacher EMA full precision/dense.
        return None

    def state_dict(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "version": 2,
            "pruner": self.pruner.state_dict(),
            "quant_ratio": float(self.quantizer.state.ratio),
            "quant_bits": int(self.quantizer.state.num_bits),
            "last_mask_update_step": int(self._last_mask_update_step),
            "current_stats": dict(self.current_stats),
        }

    def load_state_dict(self, state: dict) -> None:
        if not self.enabled or not state.get("enabled", False):
            return
        self.pruner.load_state_dict(state.get("pruner", {}))
        self.quantizer.state.num_bits = int(state.get("quant_bits", self.quantizer.state.num_bits))
        self.quantizer.set_ratio(float(state.get("quant_ratio", 0.0)))
        self._last_mask_update_step = int(state.get("last_mask_update_step", -1))
        self.current_stats = dict(state.get("current_stats", {}))

    def save_state(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)


def add_card_v2_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("CARD-v2 / reliability-guided compression")
    group.add_argument("--card_enable", action="store_true")
    group.add_argument("--card_prune_target", type=float, default=0.30)
    group.add_argument("--card_prune_interval", type=int, default=200)
    group.add_argument("--card_quant_bits", type=int, default=8)
    group.add_argument("--card_quant_target_ratio", type=float, default=1.0)
    group.add_argument("--card_delay_epochs", type=float, default=5.0, help="Explicit no-compression delay; default matches MARD warmup.")
    group.add_argument("--card_warmup_epochs", type=float, default=10.0)
    group.add_argument("--card_gate_threshold", type=float, default=0.5)
    group.add_argument(
        "--card_allow_regrowth",
        action="store_true",
        help="Allow channels to re-enter if reliability drops / masks are recomputed. Default is monotonic pruning.",
    )


# Backward-friendly aliases for scripts that want a small diff.
CardController = CardControllerV2
add_card_args = add_card_v2_args
card_ratio = reliability_ramp
