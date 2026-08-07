"""CARD: Compression-Aware Representation Diversification.

RT-SFOD-Lite "Try 2": compress the *student* network progressively, in-loop,
during source-free self-training — instead of the conventional recipe of
"train first, compress after". The teacher stays full-precision/dense at all
times and continues to act as the stable pseudo-label source (identical role
to vanilla RT-SFOD); only the student is compressed.

Two techniques are combined, both student-only, both scheduled with the same
ramp+confidence-gate shape already used by MARD's `mard_weight` in
stage2_rtsfod_yolo26.py (see Eq. 9 of the RT-SFOD paper):

  1. Progressive structured (channel) pruning
     - L1-magnitude channel importance, recomputed periodically.
     - Target sparsity ramps from 0 -> --card_prune_target over
       --card_warmup_epochs, gated by pseudo-label confidence so pruning
       only accelerates once DHF's pseudo-labels are trustworthy.
     - Implemented as a *mask* (weights zeroed, tensor shape unchanged) —
       this is a training-time simulation of structured pruning suitable for
       measuring the accuracy/sparsity trade-off. Turning the masked model
       into a physically smaller (and thus actually faster) checkpoint is a
       separate, mechanical export step (see `export_pruned_state_dict`)
       that removes zeroed filters and rewires adjacent layers.

  2. Quantization-aware training (QAT), weight-only, per-tensor symmetric
     - Straight-through estimator (STE) fake-quantization blended with the
       float weight via the same ramp/gate ratio, so training starts at full
       precision and only gradually leans on the quantized weight. This
       avoids the instability of switching to N-bit weights abruptly on top
       of already-noisy pseudo-label supervision (the same concern the paper
       raises for MARD's own warmup/gate design).

Why not just prune/quantize after Stage 2 finishes? Because pruning removes
capacity and quantization adds noise; if that happens *after* the model has
already converged on adapted pseudo-labels, the model has no more source-free
signal left to re-adapt around the induced errors (no source data, and by
then the teacher may have drifted with it via EMA). Compressing in-loop lets
the still-running self-training process absorb the extra noise the same way
it absorbs pseudo-label noise.
"""

from __future__ import annotations

import argparse
import re
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Schedule (mirrors `mard_weight` in stage2_rtsfod_yolo26.py)
# ---------------------------------------------------------------------------


def card_ratio(
    target: float,
    global_step: int,
    steps_per_epoch: int,
    warmup_epochs: float,
    avg_conf: float,
    gate_threshold: float,
) -> float:
    """Compute the current compression 'strength' in [0, target].

    Same shape as MARD's lambda(t) = lambda0 * ramp(t) * gate(avg_conf):
    linear warmup over `warmup_epochs`, then gated by pseudo-label confidence
    so compression only ramps up once DHF pseudo-labels are reliable.
    """
    warmup_steps = max(1, int(warmup_epochs * steps_per_epoch))
    ramp = min(1.0, float(global_step) / float(warmup_steps))
    gate = (avg_conf - gate_threshold) / max(1.0 - gate_threshold, 1e-6)
    gate = float(np.clip(gate, 0.0, 1.0))
    return target * ramp * gate


# ---------------------------------------------------------------------------
# 1. Progressive structured (channel) pruning
# ---------------------------------------------------------------------------

# Layer names never pruned: stem (first conv, too early / low channel count)
# and anything inside the Detect head (one2one/one2many branches, DFL) —
# pruning prediction-layer channels directly changes num classes / box dims
# and is a materially different (and much riskier) operation than pruning
# backbone/neck feature channels. This mirrors RT-SFOD's own MARD design,
# which likewise only touches PAN backbone/neck features, never head outputs.
DEFAULT_PRUNE_EXCLUDE = (r"^model\.0\.", r"model\.\d+\.(one2one|one2many)\.", r"\.dfl\.")


class ChannelPruner:
    """Magnitude-based structured channel pruning via zero-masking.

    Operates on every `nn.Conv2d` reachable from the student model whose
    qualified name does not match an exclude pattern. Masking (rather than
    physically resizing tensors) avoids having to track cross-layer channel
    dependencies during training; see `export_pruned_state_dict` for the
    deployment-time conversion to an actually-smaller checkpoint.
    """

    def __init__(self, model: nn.Module, exclude_patterns: Iterable[str] = DEFAULT_PRUNE_EXCLUDE):
        self.exclude_patterns = [re.compile(p) for p in exclude_patterns]
        self.convs: dict[str, nn.Conv2d] = {}
        self.masks: dict[str, torch.Tensor] = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and not self._is_excluded(name):
                # Skip depthwise / near-1x1-out convs where "channel" pruning
                # is degenerate (out_channels tiny, e.g. detect regressors).
                if module.out_channels < 8:
                    continue
                self.convs[name] = module
                self.masks[name] = torch.ones(module.out_channels, dtype=torch.bool, device=module.weight.device)

    def _is_excluded(self, name: str) -> bool:
        return any(p.search(name) for p in self.exclude_patterns)

    @torch.no_grad()
    def update_masks(self, target_sparsity: float) -> float:
        """Recompute per-layer masks so overall sparsity ~= target_sparsity.

        Uses global L1-magnitude ranking *within each layer* (per-layer
        uniform sparsity) rather than a single cross-layer threshold, since
        conv layers at different depths have very different weight scales.
        Returns the achieved fraction of pruned channels (for logging).
        """
        if target_sparsity <= 0.0:
            return 0.0
        total, pruned = 0, 0
        for name, conv in self.convs.items():
            importance = conv.weight.detach().abs().sum(dim=(1, 2, 3))  # (out_channels,)
            k = int(round(target_sparsity * conv.out_channels))
            k = min(k, conv.out_channels - 1)  # never prune every output channel
            mask = torch.ones_like(importance, dtype=torch.bool)
            if k > 0:
                prune_idx = torch.topk(importance, k, largest=False).indices
                mask[prune_idx] = False
            self.masks[name] = mask
            total += mask.numel()
            pruned += int((~mask).sum().item())
        self.apply_masks()
        return pruned / max(total, 1)

    @torch.no_grad()
    def apply_masks(self) -> None:
        """Zero out pruned output channels (weight + bias) in place.

        Call this after every optimizer.step() — gradient updates can move
        masked weights away from zero, so the mask must be re-applied each
        step (standard practice for mask-based pruning during training).
        """
        for name, conv in self.convs.items():
            mask = self.masks[name].to(conv.weight.device)
            if not mask.all():
                conv.weight.data[~mask] = 0.0
                if conv.bias is not None:
                    conv.bias.data[~mask] = 0.0

    def sparsity(self) -> float:
        """Current achieved fraction of pruned channels across tracked layers."""
        total = sum(m.numel() for m in self.masks.values())
        pruned = sum(int((~m).sum().item()) for m in self.masks.values())
        return pruned / max(total, 1)


@torch.no_grad()
def export_pruned_state_dict(model: nn.Module, pruner: ChannelPruner) -> dict[str, torch.Tensor]:
    """Return a state_dict with pruned output channels physically removed.

    NOTE: this is a reference/starting point, not a fully general solution.
    Removing output channels from conv `X` requires also slicing the matching
    *input* channels of every conv that directly consumes `X`'s output
    (including any BatchNorm in between, which must be sliced along its own
    channel dim). Ultralytics' Conv wrapper makes this a 1:1 mapping for the
    common case (Conv -> BN -> act -> next Conv), but concat/skip connections
    (as in C2fFaster / C2f) fan the channel indices out across the resulting
    tensor and must be tracked explicitly. Left as a follow-up utility once a
    layer-connectivity map has been extracted from the model graph — flagging
    this explicitly here rather than shipping a silently-incorrect resize.
    """
    raise NotImplementedError(
        "Physical (shape-changing) pruning export requires a layer-connectivity "
        "map for C2fFaster's concat pattern; see docstring. Training-time "
        "mask-based pruning (ChannelPruner.update_masks/apply_masks) is fully "
        "functional and sufficient for the sparsity-vs-accuracy ablation."
    )


# ---------------------------------------------------------------------------
# 2. Quantization-aware training (weight-only, per-tensor symmetric, STE)
# ---------------------------------------------------------------------------


class _FakeQuantSTE(torch.autograd.Function):
    """Symmetric per-tensor fake quantization with a straight-through gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, num_bits: int) -> torch.Tensor:
        qmax = 2 ** (num_bits - 1) - 1
        max_val = x.detach().abs().max().clamp(min=1e-8)
        scale = max_val / qmax
        x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale
        return x_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None  # STE: pass gradient through unchanged


def fake_quantize(x: torch.Tensor, num_bits: int = 8) -> torch.Tensor:
    """Simulate num_bits quantization with a straight-through gradient."""
    return _FakeQuantSTE.apply(x, num_bits)


DEFAULT_QAT_EXCLUDE = (r"^model\.0\.", r"model\.\d+\.(one2one|one2many)\.", r"\.dfl\.")


class _QATState:
    """Mutable container so the ratio/bits can be updated without re-patching forwards."""

    __slots__ = ("ratio", "num_bits")

    def __init__(self, ratio: float = 0.0, num_bits: int = 8):
        self.ratio = ratio
        self.num_bits = num_bits


def _qat_conv_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Replacement forward for nn.Conv2d: blend float and fake-quantized weight.

    Reads compression state off ``self._card_qat_state`` (set by
    `install_qat_hooks`) rather than taking it as a call argument, since the
    bound method is invoked via the normal `module(x)` / `nn.Module.__call__`
    path, which only ever forwards the tensor argument(s).
    """
    state = self._card_qat_state
    if state.ratio <= 0.0:
        w = self.weight
    else:
        w_q = fake_quantize(self.weight, state.num_bits)
        w = (1.0 - state.ratio) * self.weight + state.ratio * w_q
    return self._conv_forward(x, w, self.bias)


def install_qat_hooks(
    model: nn.Module,
    num_bits: int = 8,
    exclude_patterns: Iterable[str] = DEFAULT_QAT_EXCLUDE,
) -> tuple[_QATState, list[str]]:
    """Monkey-patch every eligible nn.Conv2d.forward to blend in fake-quant weights.

    Returns the mutable `_QATState` (update `.ratio` each step via
    `set_qat_ratio`) and the list of patched layer names, for logging.
    """
    patterns = [re.compile(p) for p in exclude_patterns]
    state = _QATState(ratio=0.0, num_bits=num_bits)
    patched = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and not any(p.search(name) for p in patterns):
            module.forward = _qat_conv_forward.__get__(module, nn.Conv2d)
            module._card_qat_state = state  # keep a reference alive on the module too
            patched.append(name)
    return state, patched


def set_qat_ratio(state: _QATState, ratio: float) -> None:
    """Update the blend ratio used by every patched Conv2d forward."""
    state.ratio = float(np.clip(ratio, 0.0, 1.0))


# ---------------------------------------------------------------------------
# CLI arg helpers (add to an argparse.ArgumentParser from the training script)
# ---------------------------------------------------------------------------


def add_card_args(parser: argparse.ArgumentParser) -> None:
    """Register CARD's CLI flags. Call from stage2_card_rtsfod_yolo26.py."""
    group = parser.add_argument_group("CARD (compression-aware self-training)")
    group.add_argument("--card_enable", action="store_true", help="Enable in-loop pruning + QAT of the student.")
    group.add_argument("--card_prune_target", type=float, default=0.3, help="Target fraction of channels pruned.")
    group.add_argument("--card_prune_interval", type=int, default=200, help="Steps between mask recomputation.")
    group.add_argument("--card_quant_bits", type=int, default=8, help="Weight bit-width for fake quantization.")
    group.add_argument("--card_quant_target_ratio", type=float, default=1.0, help="Max float->quant blend ratio.")
    group.add_argument("--card_warmup_epochs", type=float, default=10.0, help="Epochs to ramp compression to target.")
    group.add_argument(
        "--card_gate_threshold", type=float, default=0.5, help="Pseudo-label avg-confidence gate threshold."
    )


class CardController:
    """Bundles the pruner + QAT state and exposes a single `step()` call site
    to keep the training loop in stage2_card_rtsfod_yolo26.py minimal."""

    def __init__(self, student_model: nn.Module, args: argparse.Namespace):
        self.args = args
        self.pruner: Optional[ChannelPruner] = ChannelPruner(student_model) if args.card_enable else None
        self.qat_state: Optional[_QATState] = None
        if args.card_enable:
            self.qat_state, self._qat_layers = install_qat_hooks(student_model, num_bits=args.card_quant_bits)

    def step(self, global_step: int, steps_per_epoch: int, avg_conf: float) -> dict[str, float]:
        """Call once per training step (after optimizer.step()). Returns stats for logging."""
        if not self.args.card_enable:
            return {}
        prune_ratio = card_ratio(
            self.args.card_prune_target,
            global_step,
            steps_per_epoch,
            self.args.card_warmup_epochs,
            avg_conf,
            self.args.card_gate_threshold,
        )
        quant_ratio = card_ratio(
            self.args.card_quant_target_ratio,
            global_step,
            steps_per_epoch,
            self.args.card_warmup_epochs,
            avg_conf,
            self.args.card_gate_threshold,
        )
        set_qat_ratio(self.qat_state, quant_ratio)

        if global_step % self.args.card_prune_interval == 0:
            self.pruner.update_masks(prune_ratio)
        else:
            self.pruner.apply_masks()  # re-zero pruned weights every step, even between mask updates

        return {
            "card_prune_target": prune_ratio,
            "card_sparsity": self.pruner.sparsity(),
            "card_quant_ratio": quant_ratio,
        }
