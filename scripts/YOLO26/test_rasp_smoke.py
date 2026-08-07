"""Standalone smoke tests for the RASP pruning core (no Ultralytics required)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rasp_pruning import AdaptiveHiddenPruner, ChannelPack, discover_knee, export_compact_model, recursive_max_abs_diff


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c=16, hidden=32):
        super().__init__()
        self.cv1 = Conv(c, hidden, 1, True)
        self.cv2 = Conv(hidden, c, 3, False)
        self.add = True

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Conv(3, 16, 3)
        self.model = nn.Sequential(Bottleneck(16, 32), Bottleneck(16, 32))

    def forward(self, x):
        return self.model(self.stem(x))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)
    model = Toy().train()
    p = AdaptiveHiddenPruner(model, round_to=8, min_keep_ratio=0.5, gmm_min_samples=16)
    assert len(p.groups) == 2, list(p.groups)
    p.install()

    # 1) Taylor hooks collect target sensitivity without altering parameters.
    p.begin_epoch()
    x = torch.randn(2, 3, 24, 24)
    y = model(x)
    y.square().mean().backward()
    p.finalize_importance_epoch()
    assert all(g.importance_batches > 0 for g in p.groups.values())
    assert all(float(g.importance_ema.sum()) > 0 for g in p.groups.values())

    # 2) Gating is forward-only: stored latent filters stay dense.
    first_name = sorted(p.groups)[0]
    first = p.groups[first_name]
    dense_nonzero = torch.count_nonzero(first.first_conv.weight).item()
    pack = ChannelPack(first_name, tuple(range(8)), 1.0, 1.0, 0.99, 100.0, 10.0, 0.01)
    applied = p.apply_packs([pack])
    assert len(applied) == 1
    assert torch.count_nonzero(first.first_conv.weight).item() == dense_nonzero
    z = first.first(torch.randn(1, first.first_conv.in_channels, 8, 8))
    assert torch.count_nonzero(z[:, :8]).item() == 0

    # 3) Physical export is genuinely smaller and numerically equivalent.
    model.eval()
    x2 = torch.randn(1, 3, 24, 24)
    with torch.no_grad():
        y_mask = model(x2)
    state = p.state_dict()
    p.uninstall()
    dense_params = nparams(model)
    compact = export_compact_model(model, state).eval()
    compact_params = nparams(compact)
    assert compact_params < dense_params
    with torch.no_grad():
        y_compact = compact(x2)
    diff = recursive_max_abs_diff(y_mask, y_compact)
    assert diff < 2e-5, diff

    # 4) GMM can discover a low-importance component without fixed sparsity.
    model3 = Toy().eval()
    p3 = AdaptiveHiddenPruner(model3, round_to=8, min_keep_ratio=0.5, gmm_min_samples=16, seed=0)
    p3.install()
    with torch.no_grad():
        _ = model3(torch.randn(1, 3, 24, 24))  # observe feature HW / cost
    for g in p3.groups.values():
        low = torch.linspace(0.001, 0.003, 16)
        high = torch.linspace(0.5, 1.0, 16)
        g.importance_ema.copy_(torch.cat([low, high]))
    packs = p3.build_candidate_packs()
    assert len(packs) >= 2, "Expected at least two low-importance packs"
    assert all(len(pk.idxs) == 8 for pk in packs)
    p3.uninstall()

    # 5) Knee fallback selects a useful compute-vs-information point.
    xk = np.array([0.1, 0.2, 0.35, 0.5, 0.7, 1.0])
    yk = np.array([0.01, 0.03, 0.08, 0.18, 0.48, 1.0])
    idx, method = discover_knee(xk, yk)
    assert idx is not None and 1 <= idx <= 4, (idx, method)

    print("[OK] RASP smoke test")
    print(f"groups={len(state['groups'])} dense_params={dense_params} compact_params={compact_params}")
    print(f"masked_vs_compact_max_abs_diff={diff:.3e}")
    print(f"gmm_candidate_packs={len(packs)} knee_method={method} knee_index={idx}")


if __name__ == "__main__":
    main()
