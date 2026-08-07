"""RASP-SFOD: Reliability-Aware Structured Pruning for RT-SFOD / YOLO26.

This module implements the Phase-1 method selected for the project:

    Dense RT-SFOD teacher  ->  adaptively pruned student

The teacher remains dense/full precision.  The student keeps *dense latent
parameters* so parameter-wise EMA is still valid, while structured hidden
channels are zero-gated only in the student's forward pass.  Channel groups are
not pruned with a fixed per-layer percentage.  Instead the controller learns a
non-uniform pruning pattern from unlabeled target data using:

1. Dependency-safe local hidden groups (Bottleneck cv1.out -> cv2.in).
2. Target-domain Taylor importance accumulated from the actual SFOD loss.
3. Per-block GMM redundancy discovery (1-vs-2 component BIC test).
4. Cost-aware global ranking of hardware-friendly channel packs.
5. Kneedle / knee fallback to discover a data-driven pruning budget.
6. DHF reliability gating and recovery intervals between pruning events.

Why the physical pruning unit is a Bottleneck hidden channel
-------------------------------------------------------------
YOLO26 contains CSP/concat/residual/fan-out paths.  Arbitrarily masking external
feature channels is easy to simulate but difficult to turn into an *honestly
smaller* graph.  The hidden channel of a standard Bottleneck has a strictly
local dependency:

    input(C) -> cv1 -> hidden(H) -> cv2 -> output(C)

Removing hidden channel j only requires slicing cv1 output j (+ BN state) and
cv2 input j.  The block's external C->C interface does not change, so PAN,
P3/P4/P5, Detect O2O/O2M, DHF and MARD remain untouched.  Torch-Pruning's
DepGraph is used as an optional structural audit to confirm that the candidate
root group stays local to the Bottleneck.

The module deliberately contains *no quantization*.  QAT belongs to Phase 2 so
pruning can first be evaluated in isolation.
"""

from __future__ import annotations

import contextlib
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

try:  # main dependency for redundancy discovery
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover - handled at runtime
    GaussianMixture = None

try:  # optional; a deterministic geometric fallback is implemented below
    from kneed import KneeLocator
except Exception:  # pragma: no cover - handled at runtime
    KneeLocator = None


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------


@dataclass
class HiddenGroup:
    """One physically exportable Bottleneck hidden channel space."""

    name: str
    block: nn.Module
    first: nn.Module
    second: nn.Module
    first_conv: nn.Conv2d
    second_conv: nn.Conv2d
    mask: torch.Tensor
    importance_ema: torch.Tensor
    importance_sum: torch.Tensor
    importance_batches: int = 0
    output_hw: Optional[tuple[int, int]] = None
    depgraph_ok: Optional[bool] = None
    depgraph_details: str = ""

    @property
    def hidden(self) -> int:
        return int(self.mask.numel())

    @property
    def active(self) -> int:
        return int(self.mask.sum().item())

    @property
    def pruned(self) -> int:
        return self.hidden - self.active


@dataclass(frozen=True)
class ChannelPack:
    """A hardware-friendly group of hidden channels removed together."""

    block_name: str
    idxs: tuple[int, ...]
    importance: float
    normalized_importance: float
    posterior_low: float
    cost_macs: float
    cost_params: float
    score: float


@dataclass
class SelectionResult:
    packs: list[ChannelPack] = field(default_factory=list)
    knee_pack_count: int = 0
    knee_cost_fraction: float = 0.0
    candidate_pack_count: int = 0
    method: str = "none"


# -----------------------------------------------------------------------------
# Small numerical helpers
# -----------------------------------------------------------------------------


def _conv_out_dim(size: int, conv: nn.Conv2d, axis: int) -> int:
    k = conv.kernel_size[axis]
    s = conv.stride[axis]
    p = conv.padding[axis]
    d = conv.dilation[axis]
    return int(math.floor((size + 2 * p - d * (k - 1) - 1) / s + 1))


def _safe_median(x: np.ndarray, eps: float = 1e-12) -> float:
    if x.size == 0:
        return 1.0
    v = float(np.median(x))
    return max(v, eps)


def _normalized_max_distance_knee(x: np.ndarray, y: np.ndarray) -> Optional[int]:
    """Return the elbow index using max distance below the y=x diagonal.

    For our cumulative curve, x is compute removed and y is importance lost.
    Packs are sorted from cheapest-information-loss to most expensive.  A useful
    operating point maximizes x-y: much compute has been removed while little
    cumulative importance has been sacrificed.
    """

    if x.size < 2 or y.size != x.size:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    dist = x - y
    idx = int(np.argmax(dist))
    if dist[idx] <= 0:
        return None
    return idx


def discover_knee(x: Sequence[float], y: Sequence[float]) -> tuple[Optional[int], str]:
    """Discover a knee using kneed when possible, with a deterministic fallback."""

    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.size < 2:
        return None, "insufficient"

    if KneeLocator is not None:
        try:
            # y is cumulative importance loss: low initially and increasingly
            # steep as pruning reaches sensitive packs -> convex/increasing.
            kl = KneeLocator(xa, ya, curve="convex", direction="increasing", online=False)
            if kl.knee is not None:
                idx = int(np.argmin(np.abs(xa - float(kl.knee))))
                return idx, "kneed"
        except Exception:
            pass

    idx = _normalized_max_distance_knee(xa, ya)
    return idx, "max_distance" if idx is not None else "none"


# -----------------------------------------------------------------------------
# Adaptive hidden-channel pruner
# -----------------------------------------------------------------------------


class AdaptiveHiddenPruner:
    """Student-only adaptive structured pruner for standard Bottleneck blocks.

    Discovery is intentionally structural instead of importing a specific
    Ultralytics class.  An eligible block must:
      - expose ``cv1`` and ``cv2`` submodules;
      - each wrapper must expose ``.conv`` Conv2d;
      - cv1.out_channels == cv2.in_channels;
      - both convolutions use groups=1;
      - the hidden width is large enough to keep at least ``min_hidden``.

    This catches plain ``Bottleneck`` units recursively inside C3k2/C3k stages
    while excluding the Detect head and attention modules by ancestry/name.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        importance_beta: float = 0.9,
        min_hidden: int = 16,
        min_keep_ratio: float = 0.5,
        round_to: int = 8,
        gmm_posterior: float = 0.80,
        gmm_bic_gain: float = 10.0,
        gmm_min_separation: float = 1.0,
        gmm_min_samples: int = 16,
        cost_gamma: float = 1.0,
        seed: int = 29,
        exclude_name_substrings: Iterable[str] = ("one2one", "one2many", ".dfl", "c2psa", ".attn", ".psa"),
    ):
        if not 0.0 <= importance_beta < 1.0:
            raise ValueError("importance_beta must be in [0, 1)")
        if not 0.0 < min_keep_ratio <= 1.0:
            raise ValueError("min_keep_ratio must be in (0, 1]")
        self.model = model
        self.importance_beta = float(importance_beta)
        self.min_hidden = int(min_hidden)
        self.min_keep_ratio = float(min_keep_ratio)
        self.round_to = max(int(round_to), 1)
        self.gmm_posterior = float(gmm_posterior)
        self.gmm_bic_gain = float(gmm_bic_gain)
        self.gmm_min_separation = float(gmm_min_separation)
        self.gmm_min_samples = int(gmm_min_samples)
        self.cost_gamma = float(cost_gamma)
        self.seed = int(seed)
        self.exclude_name_substrings = tuple(str(x).lower() for x in exclude_name_substrings)

        self.groups: dict[str, HiddenGroup] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._installed = False
        self.collect_importance = True
        self._discover()

    # ---- discovery ---------------------------------------------------------

    def _is_excluded(self, name: str, module: nn.Module) -> bool:
        low = name.lower()
        if any(token in low for token in self.exclude_name_substrings):
            return True
        # Only standard Bottleneck-like units, not C2f/C3 containers themselves.
        if module.__class__.__name__.lower() != "bottleneck":
            return True
        return False

    def _discover(self) -> None:
        for name, module in self.model.named_modules():
            if self._is_excluded(name, module):
                continue
            first = getattr(module, "cv1", None)
            second = getattr(module, "cv2", None)
            c1 = getattr(first, "conv", None)
            c2 = getattr(second, "conv", None)
            if not isinstance(c1, nn.Conv2d) or not isinstance(c2, nn.Conv2d):
                continue
            if c1.groups != 1 or c2.groups != 1:
                continue
            if c1.out_channels != c2.in_channels:
                continue
            if c1.out_channels < self.min_hidden:
                continue
            device = c1.weight.device
            h = int(c1.out_channels)
            self.groups[name] = HiddenGroup(
                name=name,
                block=module,
                first=first,
                second=second,
                first_conv=c1,
                second_conv=c2,
                mask=torch.ones(h, dtype=torch.bool, device=device),
                importance_ema=torch.zeros(h, dtype=torch.float32, device=device),
                importance_sum=torch.zeros(h, dtype=torch.float32, device=device),
            )

    # ---- hooks / Taylor importance ----------------------------------------

    def _forward_hook(self, item: HiddenGroup):
        def hook(_module: nn.Module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.dim() < 2:
                return output
            if output.shape[1] != item.hidden:
                return output

            if output.dim() >= 4:
                item.output_hw = (int(output.shape[-2]), int(output.shape[-1]))

            # Register Taylor statistic before applying the gate.  The gradient
            # arriving at this tensor already reflects the downstream gate, so
            # permanently pruned channels naturally receive zero importance.
            if self.collect_importance and output.requires_grad:
                act = output.detach()

                def grad_hook(grad: torch.Tensor):
                    with torch.no_grad():
                        dims = [0] + list(range(2, grad.dim()))
                        imp = (act * grad.detach()).abs().mean(dim=dims).float()
                        if imp.numel() == item.hidden and torch.isfinite(imp).all():
                            item.importance_sum.add_(imp)
                            item.importance_batches += 1
                    return grad

                output.register_hook(grad_hook)

            shape = [1, item.hidden] + [1] * (output.dim() - 2)
            gate = item.mask.to(device=output.device, dtype=output.dtype).view(*shape)
            return output * gate

        return hook

    def install(self) -> None:
        if self._installed:
            return
        for item in self.groups.values():
            self._hook_handles.append(item.first.register_forward_hook(self._forward_hook(item)))
        self._installed = True

    def uninstall(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._installed = False

    @contextlib.contextmanager
    def suspended(self) -> Iterator[None]:
        """Temporarily remove gates/hooks, e.g. when serializing dense latents."""

        was_installed = self._installed
        if was_installed:
            self.uninstall()
        try:
            yield
        finally:
            if was_installed:
                self.install()

    def begin_epoch(self) -> None:
        for item in self.groups.values():
            item.importance_sum.zero_()
            item.importance_batches = 0

    @torch.no_grad()
    def finalize_importance_epoch(self) -> None:
        for item in self.groups.values():
            if item.importance_batches <= 0:
                continue
            obs = item.importance_sum / float(item.importance_batches)
            if torch.count_nonzero(item.importance_ema).item() == 0:
                item.importance_ema.copy_(obs)
            else:
                item.importance_ema.mul_(self.importance_beta).add_(obs, alpha=1.0 - self.importance_beta)

    # ---- cost model --------------------------------------------------------

    def per_channel_cost(self, item: HiddenGroup) -> tuple[float, float]:
        """Approximate MAC and parameter saving for one removed hidden channel."""

        c1, c2 = item.first_conv, item.second_conv
        k1 = int(c1.kernel_size[0] * c1.kernel_size[1])
        k2 = int(c2.kernel_size[0] * c2.kernel_size[1])
        params = float(c1.in_channels * k1 + c2.out_channels * k2)
        if c1.bias is not None:
            params += 1.0
        # Ultralytics Conv normally has affine BatchNorm after cv1.
        bn = getattr(item.first, "bn", None)
        if isinstance(bn, (nn.BatchNorm2d, nn.SyncBatchNorm)) and bn.affine:
            params += 2.0

        if item.output_hw is None:
            return params, params  # stable fallback before a shape-observation forward

        h1, w1 = item.output_hw
        h2 = _conv_out_dim(h1, c2, 0)
        w2 = _conv_out_dim(w1, c2, 1)
        macs = float(h1 * w1 * c1.in_channels * k1 + h2 * w2 * c2.out_channels * k2)
        return macs, params

    def total_prunable_cost(self) -> tuple[float, float]:
        macs = params = 0.0
        for item in self.groups.values():
            c_macs, c_params = self.per_channel_cost(item)
            macs += c_macs * item.hidden
            params += c_params * item.hidden
        return macs, params

    def current_saved_cost(self) -> tuple[float, float]:
        macs = params = 0.0
        for item in self.groups.values():
            c_macs, c_params = self.per_channel_cost(item)
            macs += c_macs * item.pruned
            params += c_params * item.pruned
        return macs, params

    # ---- GMM redundancy discovery ----------------------------------------

    def _block_channel_candidates(self, item: HiddenGroup) -> list[tuple[int, float, float, float, float]]:
        """Return (idx, raw_imp, norm_imp, posterior_low, channel_macs) candidates."""

        if GaussianMixture is None:
            raise RuntimeError("scikit-learn is required for RASP GMM selection: pip install scikit-learn")

        active_idx = torch.where(item.mask)[0].detach().cpu().numpy().astype(np.int64)
        if active_idx.size < max(self.gmm_min_samples, 2 * self.round_to):
            return []

        imp_all = item.importance_ema.detach().float().cpu().numpy()
        imp = imp_all[active_idx]
        if not np.isfinite(imp).all() or float(np.max(imp)) <= 0:
            return []

        eps = max(float(np.max(imp)) * 1e-8, 1e-12)
        x = np.log(imp + eps).reshape(-1, 1)

        try:
            g1 = GaussianMixture(n_components=1, covariance_type="full", n_init=3, random_state=self.seed).fit(x)
            g2 = GaussianMixture(n_components=2, covariance_type="full", n_init=5, random_state=self.seed).fit(x)
        except Exception:
            return []

        bic_gain = float(g1.bic(x) - g2.bic(x))
        means = g2.means_.reshape(-1)
        vars_ = g2.covariances_.reshape(-1)
        low = int(np.argmin(means))
        high = 1 - low
        separation = abs(float(means[high] - means[low])) / math.sqrt(max(float(vars_[low] + vars_[high]), 1e-12))
        if bic_gain < self.gmm_bic_gain or separation < self.gmm_min_separation:
            return []

        probs = g2.predict_proba(x)[:, low]
        median = _safe_median(imp)
        c_macs, _ = self.per_channel_cost(item)

        max_prunable = max(0, item.hidden - max(self.min_hidden, int(math.ceil(item.hidden * self.min_keep_ratio))))
        remaining_allowance = max(0, max_prunable - item.pruned)
        if remaining_allowance < self.round_to:
            return []

        rows = []
        for pos, idx in enumerate(active_idx.tolist()):
            p = float(probs[pos])
            if p < self.gmm_posterior:
                continue
            raw = float(imp[pos])
            rows.append((int(idx), raw, raw / median, p, c_macs))

        # GMM can occasionally mark more channels than the block safety limit.
        rows.sort(key=lambda r: (r[2] / max(r[3], 1e-6), r[0]))
        return rows[:remaining_allowance]

    def build_candidate_packs(self) -> list[ChannelPack]:
        channel_rows: dict[str, list[tuple[int, float, float, float, float]]] = {}
        all_costs = []
        for name, item in self.groups.items():
            rows = self._block_channel_candidates(item)
            if rows:
                channel_rows[name] = rows
                all_costs.extend(r[4] for r in rows)

        if not all_costs:
            return []
        cost_ref = _safe_median(np.asarray(all_costs, dtype=np.float64))

        packs: list[ChannelPack] = []
        for name, rows in channel_rows.items():
            item = self.groups[name]
            # Rank within a block, then prune in complete round_to packs so the
            # compact hidden width remains accelerator-friendly.
            scored_rows = []
            for idx, raw, norm, p, macs in rows:
                cost_rel = max(macs / cost_ref, 1e-12)
                score = norm / (max(p, 1e-6) * (cost_rel ** self.cost_gamma))
                scored_rows.append((score, idx, raw, norm, p, macs))
            scored_rows.sort(key=lambda x: (x[0], x[1]))

            usable = (len(scored_rows) // self.round_to) * self.round_to
            for start in range(0, usable, self.round_to):
                chunk = scored_rows[start : start + self.round_to]
                idxs = tuple(sorted(int(r[1]) for r in chunk))
                raw_imp = float(sum(r[2] for r in chunk))
                norm_imp = float(sum(r[3] for r in chunk))
                p_low = float(np.mean([r[4] for r in chunk]))
                macs = float(sum(r[5] for r in chunk))
                _, c_params = self.per_channel_cost(item)
                params = float(c_params * len(chunk))
                cost_rel = max((macs / len(chunk)) / cost_ref, 1e-12)
                score = (norm_imp / len(chunk)) / (max(p_low, 1e-6) * (cost_rel ** self.cost_gamma))
                packs.append(
                    ChannelPack(
                        block_name=name,
                        idxs=idxs,
                        importance=raw_imp,
                        normalized_importance=norm_imp,
                        posterior_low=p_low,
                        cost_macs=macs,
                        cost_params=params,
                        score=float(score),
                    )
                )

        packs.sort(key=lambda p: (p.score, p.block_name, p.idxs))
        return packs

    # ---- Kneedle budget ----------------------------------------------------

    def select_knee_packs(self, packs: list[ChannelPack]) -> SelectionResult:
        if not packs:
            return SelectionResult(candidate_pack_count=0)
        total_cost = max(sum(p.cost_macs for p in packs), 1e-12)
        total_imp = max(sum(p.normalized_importance for p in packs), 1e-12)
        x = np.cumsum([p.cost_macs for p in packs], dtype=np.float64) / total_cost
        y = np.cumsum([p.normalized_importance for p in packs], dtype=np.float64) / total_imp
        idx, method = discover_knee(x, y)
        if idx is None:
            return SelectionResult(candidate_pack_count=len(packs), method=method)
        n = int(idx) + 1
        return SelectionResult(
            packs=packs[:n],
            knee_pack_count=n,
            knee_cost_fraction=float(x[idx]),
            candidate_pack_count=len(packs),
            method=method,
        )

    # ---- applying monotonic masks ----------------------------------------

    @torch.no_grad()
    def apply_packs(self, packs: Sequence[ChannelPack], max_new_macs: Optional[float] = None) -> list[ChannelPack]:
        selected: list[ChannelPack] = []
        used_macs = 0.0
        for pack in packs:
            item = self.groups.get(pack.block_name)
            if item is None:
                continue
            idx = torch.as_tensor(pack.idxs, dtype=torch.long, device=item.mask.device)
            if idx.numel() == 0 or not bool(item.mask[idx].all()):
                continue
            if max_new_macs is not None and selected and used_macs + pack.cost_macs > max_new_macs:
                break

            # Re-check block safety at the moment of application.
            future_active = item.active - int(idx.numel())
            min_keep = max(self.min_hidden, int(math.ceil(item.hidden * self.min_keep_ratio)))
            if future_active < min_keep:
                continue
            item.mask[idx] = False
            used_macs += float(pack.cost_macs)
            selected.append(pack)
        return selected

    # ---- reporting / state ------------------------------------------------

    def sparsity(self) -> float:
        total = sum(item.hidden for item in self.groups.values())
        pruned = sum(item.pruned for item in self.groups.values())
        return float(pruned / max(total, 1))

    def block_sparsities(self) -> dict[str, float]:
        return {name: float(item.pruned / max(item.hidden, 1)) for name, item in self.groups.items()}

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "config": {
                "importance_beta": self.importance_beta,
                "min_hidden": self.min_hidden,
                "min_keep_ratio": self.min_keep_ratio,
                "round_to": self.round_to,
                "gmm_posterior": self.gmm_posterior,
                "gmm_bic_gain": self.gmm_bic_gain,
                "gmm_min_separation": self.gmm_min_separation,
                "gmm_min_samples": self.gmm_min_samples,
                "cost_gamma": self.cost_gamma,
                "seed": self.seed,
            },
            "groups": {
                name: {
                    "mask": item.mask.detach().cpu(),
                    "importance_ema": item.importance_ema.detach().cpu(),
                    "output_hw": item.output_hw,
                    "depgraph_ok": item.depgraph_ok,
                    "depgraph_details": item.depgraph_details,
                }
                for name, item in self.groups.items()
            },
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        saved = state.get("groups", {})
        missing = []
        for name, item in self.groups.items():
            if name not in saved:
                missing.append(name)
                continue
            src = saved[name]
            mask = torch.as_tensor(src["mask"], dtype=torch.bool, device=item.mask.device)
            imp = torch.as_tensor(src.get("importance_ema", torch.zeros_like(mask, dtype=torch.float32)), dtype=torch.float32, device=item.mask.device)
            if mask.numel() != item.hidden or imp.numel() != item.hidden:
                raise ValueError(f"RASP state shape mismatch for {name}: saved={mask.numel()} current={item.hidden}")
            item.mask.copy_(mask)
            item.importance_ema.copy_(imp)
            hw = src.get("output_hw")
            item.output_hw = tuple(hw) if hw is not None else None
            item.depgraph_ok = src.get("depgraph_ok")
            item.depgraph_details = str(src.get("depgraph_details", ""))
        if strict and missing:
            raise KeyError(f"RASP state missing {len(missing)} groups, e.g. {missing[:3]}")

    # ---- optional DepGraph audit ------------------------------------------

    def audit_depgraph(self, example_inputs: torch.Tensor, require_local: bool = True) -> dict[str, dict]:
        """Audit candidate roots with Torch-Pruning DepGraph.

        This is an *audit*, not the training-time masking mechanism.  The method
        checks that pruning one hidden output channel of cv1 produces a valid
        dependency group and, when ``require_local`` is True, that all named
        parameterized modules touched by the group remain inside the Bottleneck.
        """

        try:
            import torch_pruning as tp
        except Exception as exc:  # pragma: no cover - dependency-specific
            raise RuntimeError("DepGraph audit requires `pip install torch-pruning`") from exc

        # DepGraph needs autograd enabled.  Remove RASP forward hooks so its
        # graph reflects the native dense model.
        module_names = {id(m): n for n, m in self.model.named_modules()}
        results: dict[str, dict] = {}
        with self.suspended():
            was_training = self.model.training
            self.model.eval()
            with torch.enable_grad():
                DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=example_inputs)
            if was_training:
                self.model.train()

            for name, item in self.groups.items():
                try:
                    group = DG.get_pruning_group(item.first_conv, tp.prune_conv_out_channels, idxs=[0])
                    valid = bool(DG.check_pruning_group(group))
                    outside = []
                    touched = []
                    for dep, _idxs in group:
                        target = dep.target.module
                        target_name = module_names.get(id(target))
                        if target_name is None:
                            continue  # synthetic concat/add/autograd node
                        touched.append(target_name)
                        if require_local and not (target_name == name or target_name.startswith(name + ".")):
                            outside.append(target_name)
                    ok = valid and (not outside)
                    details = str(group.details() if hasattr(group, "details") else group)
                except Exception as exc:
                    ok = False
                    valid = False
                    outside = []
                    touched = []
                    details = f"ERROR: {type(exc).__name__}: {exc}"
                item.depgraph_ok = bool(ok)
                item.depgraph_details = details
                results[name] = {
                    "ok": bool(ok),
                    "valid": bool(valid),
                    "outside_named_modules": outside,
                    "touched_named_modules": touched,
                    "details": details,
                }
        return results


# -----------------------------------------------------------------------------
# RASP controller: reliability gate + iterative pruning cycles
# -----------------------------------------------------------------------------


class RASPController:
    """High-level RASP pruning controller used by the Stage-2 training loop."""

    def __init__(self, model: nn.Module, args):
        self.model = model
        self.args = args
        self.enabled = bool(getattr(args, "rasp_enable", False))
        self.last_prune_epoch = -10**9
        self.prune_events = 0
        self.current_stats: dict[str, float | int | str] = {}
        self.pruner: Optional[AdaptiveHiddenPruner] = None

        if self.enabled:
            self.pruner = AdaptiveHiddenPruner(
                model,
                importance_beta=args.rasp_importance_beta,
                min_hidden=args.rasp_min_hidden,
                min_keep_ratio=args.rasp_min_keep_ratio,
                round_to=args.rasp_round_to,
                gmm_posterior=args.rasp_gmm_posterior,
                gmm_bic_gain=args.rasp_gmm_bic_gain,
                gmm_min_separation=args.rasp_gmm_min_separation,
                gmm_min_samples=args.rasp_gmm_min_samples,
                cost_gamma=args.rasp_cost_gamma,
                seed=getattr(args, "seed", 29),
            )
            if not self.pruner.groups:
                raise RuntimeError(
                    "RASP found no eligible Bottleneck hidden groups. Verify the YOLO26 fork/architecture before training."
                )
            self.pruner.install()

    def begin_epoch(self) -> None:
        if self.enabled:
            self.pruner.begin_epoch()

    @contextlib.contextmanager
    def suspended(self) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        with self.pruner.suspended():
            yield

    def end_epoch(self, epoch: int, reliability: float) -> dict[str, float | int | str]:
        if not self.enabled:
            return {}

        self.pruner.finalize_importance_epoch()
        total_macs, total_params = self.pruner.total_prunable_cost()
        saved_macs, saved_params = self.pruner.current_saved_cost()
        stats: dict[str, float | int | str] = {
            "rasp_reliability": float(reliability),
            "rasp_hidden_sparsity": self.pruner.sparsity(),
            "rasp_saved_prunable_macs_frac": float(saved_macs / max(total_macs, 1e-12)),
            "rasp_saved_prunable_params_frac": float(saved_params / max(total_params, 1e-12)),
            "rasp_new_packs": 0,
            "rasp_candidates": 0,
            "rasp_knee_packs": 0,
            "rasp_knee_method": "none",
        }

        warmup_done = int(epoch) > int(self.args.rasp_warmup_epochs)
        interval_done = int(epoch) - int(self.last_prune_epoch) >= int(self.args.rasp_cycle_epochs)
        reliable = float(reliability) >= float(self.args.rasp_reliability_threshold)
        if not (warmup_done and interval_done and reliable):
            reason = "warmup" if not warmup_done else "recovery_interval" if not interval_done else "low_reliability"
            stats["rasp_status"] = reason
            self.current_stats = stats
            return dict(stats)

        packs = self.pruner.build_candidate_packs()
        selection = self.pruner.select_knee_packs(packs)
        stats["rasp_candidates"] = int(selection.candidate_pack_count)
        stats["rasp_knee_packs"] = int(selection.knee_pack_count)
        stats["rasp_knee_method"] = selection.method
        stats["rasp_knee_cost_fraction"] = float(selection.knee_cost_fraction)
        if not selection.packs:
            stats["rasp_status"] = "no_safe_knee"
            self.current_stats = stats
            return dict(stats)

        # Even a data-driven knee can be large.  The step cap does not choose
        # final sparsity; it only makes the route to the knee progressive so the
        # ongoing RT-SFOD optimization can recover between pruning events.
        max_new_macs = float(self.args.rasp_max_step_cost_fraction) * max(total_macs, 1e-12)
        applied = self.pruner.apply_packs(selection.packs, max_new_macs=max_new_macs)
        if applied:
            self.last_prune_epoch = int(epoch)
            self.prune_events += 1
        new_macs = float(sum(p.cost_macs for p in applied))
        new_params = float(sum(p.cost_params for p in applied))
        saved_macs, saved_params = self.pruner.current_saved_cost()
        stats.update(
            {
                "rasp_new_packs": len(applied),
                "rasp_new_macs": new_macs,
                "rasp_new_params": new_params,
                "rasp_hidden_sparsity": self.pruner.sparsity(),
                "rasp_saved_prunable_macs_frac": float(saved_macs / max(total_macs, 1e-12)),
                "rasp_saved_prunable_params_frac": float(saved_params / max(total_params, 1e-12)),
                "rasp_prune_events": int(self.prune_events),
                "rasp_status": "pruned" if applied else "step_cap_or_safety",
            }
        )
        self.current_stats = stats
        return dict(stats)

    def state_dict(self) -> dict:
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "version": 1,
            "last_prune_epoch": int(self.last_prune_epoch),
            "prune_events": int(self.prune_events),
            "current_stats": dict(self.current_stats),
            "pruner": self.pruner.state_dict(),
        }

    def load_state_dict(self, state: dict, strict: bool = True) -> None:
        if not self.enabled:
            if strict and state.get("enabled", False):
                raise RuntimeError("Checkpoint contains RASP state but --rasp_enable is not set")
            return
        if not state.get("enabled", False):
            if strict:
                raise RuntimeError("--rasp_enable was set but checkpoint has no enabled RASP state")
            return
        self.last_prune_epoch = int(state.get("last_prune_epoch", -10**9))
        self.prune_events = int(state.get("prune_events", 0))
        self.current_stats = dict(state.get("current_stats", {}))
        self.pruner.load_state_dict(state["pruner"], strict=strict)



def pruner_from_state(model: nn.Module, pruner_state: dict) -> AdaptiveHiddenPruner:
    """Recreate an AdaptiveHiddenPruner with the exact discovery config saved in state."""

    cfg = dict(pruner_state.get("config", {}))
    allowed = {
        "importance_beta",
        "min_hidden",
        "min_keep_ratio",
        "round_to",
        "gmm_posterior",
        "gmm_bic_gain",
        "gmm_min_separation",
        "gmm_min_samples",
        "cost_gamma",
        "seed",
    }
    kwargs = {k: v for k, v in cfg.items() if k in allowed}
    return AdaptiveHiddenPruner(model, **kwargs)

# -----------------------------------------------------------------------------
# Exact physical compaction for the local hidden groups
# -----------------------------------------------------------------------------


def _new_conv_like(old: nn.Conv2d, in_channels: int, out_channels: int) -> nn.Conv2d:
    if old.groups != 1:
        raise ValueError("RASP physical compaction currently supports groups=1 only")
    new = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=1,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    ).to(device=old.weight.device, dtype=old.weight.dtype)
    return new


def _new_bn_like(old: nn.Module, num_features: int) -> nn.BatchNorm2d:
    if not isinstance(old, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        raise TypeError(f"Expected BatchNorm2d-like module, got {type(old).__name__}")
    device = old.weight.device if old.affine else old.running_mean.device
    dtype = old.weight.dtype if old.affine else old.running_mean.dtype
    new = nn.BatchNorm2d(
        num_features,
        eps=old.eps,
        momentum=old.momentum,
        affine=old.affine,
        track_running_stats=old.track_running_stats,
    ).to(device=device, dtype=dtype)
    return new


@torch.no_grad()
def _compact_hidden_group(item: HiddenGroup) -> None:
    keep = torch.where(item.mask)[0]
    if keep.numel() == item.hidden:
        return
    if keep.numel() == 0:
        raise RuntimeError(f"Cannot compact all hidden channels in {item.name}")

    first, second = item.first, item.second
    old1, old2 = item.first_conv, item.second_conv
    new_first = copy.deepcopy(first)
    new_second = copy.deepcopy(second)

    conv1 = _new_conv_like(old1, old1.in_channels, int(keep.numel()))
    conv1.weight.copy_(old1.weight[keep])
    if old1.bias is not None:
        conv1.bias.copy_(old1.bias[keep])
    new_first.conv = conv1

    old_bn = getattr(first, "bn", None)
    if isinstance(old_bn, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        bn = _new_bn_like(old_bn, int(keep.numel()))
        if old_bn.affine:
            bn.weight.copy_(old_bn.weight[keep])
            bn.bias.copy_(old_bn.bias[keep])
        if old_bn.track_running_stats:
            bn.running_mean.copy_(old_bn.running_mean[keep])
            bn.running_var.copy_(old_bn.running_var[keep])
            bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
        new_first.bn = bn

    conv2 = _new_conv_like(old2, int(keep.numel()), old2.out_channels)
    conv2.weight.copy_(old2.weight[:, keep])
    if old2.bias is not None:
        conv2.bias.copy_(old2.bias)
    new_second.conv = conv2
    # cv2 BN/output width is unchanged and is already correct in deepcopy.

    item.block.cv1 = new_first
    item.block.cv2 = new_second


def export_compact_model(model: nn.Module, rasp_state: dict) -> nn.Module:
    """Return a physically smaller model exactly matching the accepted masks."""

    compact = copy.deepcopy(model)
    state = rasp_state.get("pruner", rasp_state)
    p = pruner_from_state(compact, state)
    p.load_state_dict(state, strict=True)
    # No forward gates are installed on the copy.  Replace modules physically.
    for item in list(p.groups.values()):
        _compact_hidden_group(item)
    return compact


# -----------------------------------------------------------------------------
# Audit/report helpers
# -----------------------------------------------------------------------------


def recursive_max_abs_diff(a, b) -> float:
    """Maximum absolute tensor difference across nested YOLO output structures."""

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            return float("inf")
        if a.numel() == 0:
            return 0.0
        return float((a.detach().float() - b.detach().float()).abs().max().item())
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return float("inf")
        return max((recursive_max_abs_diff(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((recursive_max_abs_diff(x, y) for x, y in zip(a, b)), default=0.0)
    return 0.0 if type(a) is type(b) else float("inf")


def save_audit_json(path: str | Path, audit: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")


def add_rasp_args(parser) -> None:
    """Add RASP CLI flags to an argparse parser."""

    g = parser.add_argument_group("RASP adaptive structured pruning")
    g.add_argument("--rasp_enable", action="store_true", help="Enable student-only RASP structured pruning")
    g.add_argument("--rasp_warmup_epochs", type=int, default=5, help="Collect target importance before first pruning")
    g.add_argument("--rasp_cycle_epochs", type=int, default=3, help="Minimum recovery/adaptation epochs between pruning events")
    g.add_argument("--rasp_reliability_threshold", type=float, default=0.50, help="Minimum mean DHF confidence required to prune")
    g.add_argument("--rasp_importance_beta", type=float, default=0.90, help="EMA beta for target Taylor importance")
    g.add_argument("--rasp_min_hidden", type=int, default=16, help="Absolute minimum hidden channels kept per Bottleneck")
    g.add_argument("--rasp_min_keep_ratio", type=float, default=0.50, help="Per-block minimum fraction of original hidden channels kept")
    g.add_argument("--rasp_round_to", type=int, default=8, help="Prune hidden channels in accelerator-friendly packs")
    g.add_argument("--rasp_gmm_posterior", type=float, default=0.80, help="Minimum posterior probability of low-importance GMM component")
    g.add_argument("--rasp_gmm_bic_gain", type=float, default=10.0, help="Minimum BIC(1)-BIC(2) evidence before accepting a two-component split")
    g.add_argument("--rasp_gmm_min_separation", type=float, default=1.0, help="Minimum normalized separation between GMM means")
    g.add_argument("--rasp_gmm_min_samples", type=int, default=16, help="Minimum active channels needed to fit a block GMM")
    g.add_argument("--rasp_cost_gamma", type=float, default=1.0, help="Strength of compute saving in global cost-aware ranking")
    g.add_argument("--rasp_max_step_cost_fraction", type=float, default=0.05, help="Maximum fraction of baseline prunable MACs removed in one pruning event")
    g.add_argument("--rasp_depgraph_audit", action="store_true", help="Run Torch-Pruning DepGraph audit once before training")
    g.add_argument("--rasp_require_depgraph", action="store_true", help="Abort if any candidate fails the local DepGraph audit")
    g.add_argument("--rasp_audit_imgsz", type=int, default=256, help="Dummy image size used for DepGraph audit")
