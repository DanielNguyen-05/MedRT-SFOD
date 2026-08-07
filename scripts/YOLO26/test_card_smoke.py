"""Dependency-light smoke tests for CARD-v2 conceptual invariants."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scripts.YOLO26.card_compression import CardControllerV2, reliability_ramp


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Deliberately use a ModuleList named model to mimic Ultralytics enough
        # for automatic stem/head exclusion.
        self.model = nn.ModuleList([
            ConvBNAct(3, 16),
            ConvBNAct(16, 24),
            ConvBNAct(24, 32),
            ConvBNAct(32, 16),  # treated as "head" and excluded
        ])

    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        card_enable=True,
        card_prune_target=0.50,
        card_prune_interval=1,
        card_quant_bits=8,
        card_quant_target_ratio=1.0,
        card_delay_epochs=0.0,
        card_warmup_epochs=0.01,
        card_gate_threshold=0.5,
        card_allow_regrowth=False,
    )


def ema_update(teacher: nn.Module, student: nn.Module, mu: float = 0.9):
    with torch.no_grad():
        for pt, ps in zip(teacher.parameters(), student.parameters()):
            assert pt.shape == ps.shape
            pt.mul_(mu).add_(ps, alpha=1.0 - mu)


def main():
    torch.manual_seed(7)
    student = TinyNet().train()
    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    controller = CardControllerV2(student, make_args())
    assert len(controller.pruner.layers) > 0

    # Schedule sanity.
    assert reliability_ramp(0.5, 0, 100, 10, 0.9, 0.5, 5) == 0.0
    r = reliability_ramp(0.5, 2000, 100, 10, 0.9, 0.5, 5)
    assert 0.0 < r <= 0.5

    # Force full scheduled strength and inspect the first gated layer.
    stats = controller.prepare_step(optimizer_step=100, steps_per_epoch=10, reliability=1.0)
    assert stats["card_sparsity"] > 0.0
    first = next(iter(controller.pruner.layers.values()))
    pruned = ~first.mask
    assert pruned.any()

    # Critical invariant #1: latent stored weights are NOT zeroed by pruning.
    latent_before = first.conv.weight.detach().clone()
    assert torch.count_nonzero(latent_before[pruned]).item() > 0

    # Critical invariant #2: post-BN feature channels are exactly gated to zero.
    captured = {}
    def _capture(_m, _i, o):
        captured["y"] = o.detach().clone()
        return None
    handle = first.wrapper.register_forward_hook(_capture)
    x = torch.randn(2, 3, 32, 32)
    out = student(x)
    handle.remove()
    y = captured["y"]
    assert torch.allclose(y[:, pruned], torch.zeros_like(y[:, pruned]), atol=0, rtol=0)

    # Quantized/gated forward still backpropagates finite gradients.
    loss = out.square().mean()
    loss.backward()
    assert first.conv.weight.grad is not None
    assert torch.isfinite(first.conv.weight.grad).all()

    opt = torch.optim.SGD(student.parameters(), lr=1e-3, weight_decay=1e-4)
    opt.step()
    controller.after_optimizer_step()

    # Critical invariant #3: teacher EMA receives dense latent tensors, not
    # zero-baked masks.  At least one pruned latent filter remains non-zero.
    ema_update(teacher, student)
    t_layer = dict(teacher.named_modules())[first.name]
    assert torch.count_nonzero(t_layer.conv.weight.detach()[pruned]).item() > 0

    # Patches can be removed for full-model serialization and restored.
    with controller.suspended_for_serialization():
        _ = student(x)
    _ = student(x)

    state = controller.state_dict()
    assert state["version"] == 2 and state["enabled"]
    print("[OK] CARD-v2: dense-latent EMA, post-BN gating, fake-QAT gradients, serialization suspension")
    print("[OK] stats:", stats)


if __name__ == "__main__":
    main()
