# RT-SFOD extension — AGPL-3.0 License (inherits from Ultralytics)
"""Lightweight backbone blocks for the RT-SFOD-Lite extension.

This module implements Partial Convolution (PConv) blocks in the style of
FasterNet (Chen et al., CVPR'23), and wraps them in the same CSP ("C2f-style")
container used throughout Ultralytics so they are a drop-in replacement for
C2f / C3k2 stages in a model YAML.

Design goals (see RT-SFOD-Lite proposal, "Try 1"):
  * Reduce backbone FLOPs/params without shrinking the number of PAN output
    stages (P3/P4/P5 stay intact) so DHF (dual-head fusion) and MARD
    (multi-scale representation diversification) require *no* changes.
  * Target the actual latency bottleneck (memory access), not just raw FLOPs:
    PConv only convolves a fraction of the channels and leaves the rest
    untouched, which is the mechanism FasterNet uses to cut memory traffic
    while keeping representational capacity via the pointwise MLP that
    follows.

Both classes below follow the exact constructor signature convention used by
C2f/C3k2 (`c1, c2, n=1, shortcut=True, g=1, e=0.5`) so they can be registered
in `base_modules` / `repeat_modules` inside `ultralytics/nn/tasks.py` and used
directly from a YAML file, e.g.:

    - [-1, 2, C2fFaster, [256, True]]   # same call convention as C3k2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv

__all__ = ("PConv", "FasterNetBlock", "C2fFaster")


class PConv(nn.Module):
    """Partial convolution: apply a k x k conv to only 1/n_div of the channels.

    The remaining channels are passed through untouched. This is the core
    memory-access-efficient operator from FasterNet (Chen et al., CVPR 2023),
    "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks".
    """

    def __init__(self, dim: int, n_div: int = 4, k: int = 3):
        """Initialize PConv.

        Args:
            dim (int): Number of input/output channels (in-place op).
            n_div (int): Partition ratio; 1/n_div channels are convolved.
            k (int): Kernel size of the partial convolution.
        """
        super().__init__()
        self.dim_conv = max(dim // n_div, 1)
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, k, 1, k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split -> convolve first slice -> concat back with untouched slice."""
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        return torch.cat((x1, x2), dim=1)


class FasterNetBlock(nn.Module):
    """PConv + inverted-MLP (1x1 expand -> act -> 1x1 project) residual block.

    This is the elementary building block of FasterNet, adapted to reuse
    Ultralytics' `Conv` (conv+BN+act) for the pointwise layers so it behaves
    consistently with the rest of the codebase (fusable at export time).
    """

    def __init__(self, dim: int, n_div: int = 4, mlp_ratio: float = 2.0, shortcut: bool = True):
        """Initialize a FasterNet residual block.

        Args:
            dim (int): Channels in/out (block is shape-preserving).
            n_div (int): Partial-conv partition ratio, see `PConv`.
            mlp_ratio (float): Expansion ratio of the pointwise MLP.
            shortcut (bool): Whether to add the residual connection.
        """
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.pconv = PConv(dim, n_div=n_div, k=3)
        self.mlp = nn.Sequential(
            Conv(dim, hidden, k=1, act=True),
            Conv(hidden, dim, k=1, act=False),
        )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with optional residual add."""
        y = self.mlp(self.pconv(x))
        return x + y if self.add else y


class C2fFaster(nn.Module):
    """CSP-style container (same pattern as `C2f`) using `FasterNetBlock`
    instead of the standard 3x3 `Bottleneck` as its internal repeated unit.

    Drop-in replacement for `C2f` / `C3k2` in a backbone YAML: same
    constructor signature `(c1, c2, n, shortcut, g, e)` (g is accepted for
    interface compatibility but unused, since PConv does not use groups).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        n_div: int = 4,
        mlp_ratio: float = 2.0,
    ):
        """Initialize C2fFaster.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of FasterNetBlock repeats.
            shortcut (bool): Whether internal blocks use residual add.
            g (int): Unused, kept for CSP-family interface parity.
            e (float): Hidden-channel expansion ratio (as in C2f).
            n_div (int): PConv partition ratio.
            mlp_ratio (float): MLP expansion ratio inside FasterNetBlock.
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            FasterNetBlock(self.c, n_div=n_div, mlp_ratio=mlp_ratio, shortcut=shortcut) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CSP forward: split, run repeated blocks, concat, fuse."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
