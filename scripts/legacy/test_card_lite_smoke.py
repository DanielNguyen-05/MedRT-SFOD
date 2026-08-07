"""Smoke test for RT-SFOD-Lite (Try 1 backbone swap + Try 2 CARD compression).

Does NOT require the full `ultralytics` package to be importable (it builds
`DetectionModel` directly, sidestepping `from ultralytics import YOLO`, which
currently fails on this repo because `ultralytics/data/` is missing from the
public GitHub release — unrelated to the RT-SFOD-Lite changes here).

Run:
    python scripts/YOLO26/test_card_lite_smoke.py

Checks:
  1. yolo26-lite.yaml builds and forward-passes in both train() and eval()
     mode, producing the same {one2one, one2many} dual-head structure the
     DHF/MARD code in stage2_rtsfod_yolo26.py expects (boxes/scores/feats,
     3-scale PAN features).
  2. card_ratio's ramp+confidence-gate schedule behaves monotonically.
  3. ChannelPruner actually zeroes the targeted fraction of output channels
     and the model still forward-passes afterwards.
  4. QAT hooks blend float/fake-quantized weights, gradients still flow
     (STE), and ratio=0 vs ratio=1 produce measurably different outputs.
  5. CardController (the single call site stage2_card_rtsfod_yolo26.py uses)
     runs for a few optimizer steps without shape/dtype errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402

from scripts.legacy.card_compression import (  # noqa: E402
    CardController,
    ChannelPruner,
    add_card_args,
    card_ratio,
    install_qat_hooks,
    set_qat_ratio,
)

LITE_CFG = str(REPO_ROOT / "ultralytics" / "cfg" / "models" / "26" / "yolo26-lite.yaml")


def build_model(nc: int = 8) -> DetectionModel:
    torch.manual_seed(0)
    return DetectionModel(cfg=LITE_CFG, ch=3, nc=nc, verbose=False)


def check_forward_shapes() -> None:
    model = build_model()
    x = torch.randn(2, 3, 256, 256)

    model.train()
    out_train = model(x)
    assert isinstance(out_train, dict) and "one2one" in out_train and "one2many" in out_train
    assert out_train["one2one"]["boxes"].shape[0] == 2
    assert len(out_train["one2one"]["feats"]) == 3, "MARD needs exactly 3 PAN scales (P3/P4/P5)"

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert isinstance(out_eval, tuple) and len(out_eval) == 2 and isinstance(out_eval[1], dict)
    print("[OK] yolo26-lite.yaml forward pass (train + eval) matches expected DHF/MARD interface")


def check_card_schedule() -> None:
    r_cold = card_ratio(0.3, global_step=0, steps_per_epoch=100, warmup_epochs=10, avg_conf=0.3, gate_threshold=0.5)
    r_low_conf = card_ratio(
        0.3, global_step=500, steps_per_epoch=100, warmup_epochs=10, avg_conf=0.3, gate_threshold=0.5
    )
    r_gated = card_ratio(0.3, global_step=500, steps_per_epoch=100, warmup_epochs=10, avg_conf=0.6, gate_threshold=0.5)
    r_full = card_ratio(0.3, global_step=2000, steps_per_epoch=100, warmup_epochs=10, avg_conf=0.9, gate_threshold=0.5)
    assert r_cold == 0.0
    assert r_low_conf == 0.0, "below gate_threshold confidence must fully suppress compression"
    assert 0.0 < r_gated < r_full <= 0.3
    print(f"[OK] card_ratio schedule: cold={r_cold:.3f} low_conf={r_low_conf:.3f} gated={r_gated:.3f} full={r_full:.3f}")


def check_pruning() -> None:
    model = build_model()
    pruner = ChannelPruner(model)
    assert len(pruner.convs) > 0, "no prunable Conv2d layers found — check exclude patterns"
    achieved = pruner.update_masks(target_sparsity=0.3)
    assert abs(achieved - 0.3) < 0.02

    name0 = next(iter(pruner.convs))
    conv0, mask0 = pruner.convs[name0], pruner.masks[name0]
    assert torch.allclose(conv0.weight.data[~mask0], torch.zeros_like(conv0.weight.data[~mask0]))

    x = torch.randn(2, 3, 256, 256)
    model.train()
    out = model(x)  # must not raise after pruning
    assert out["one2one"]["boxes"].shape[0] == 2
    print(f"[OK] ChannelPruner: {len(pruner.convs)} prunable layers, achieved_sparsity={achieved:.3f}")


def check_qat() -> None:
    model = build_model()
    model.train()
    x = torch.randn(2, 3, 256, 256)

    qat_state, patched = install_qat_hooks(model, num_bits=8)
    assert len(patched) > 0

    set_qat_ratio(qat_state, 0.5)
    out = model(x)
    loss = out["one2one"]["boxes"].float().sum() + out["one2many"]["scores"].float().sum()
    loss.backward()
    assert model.model[0].conv.weight.grad is not None
    assert torch.isfinite(model.model[0].conv.weight.grad).all(), "STE must keep gradients finite"

    set_qat_ratio(qat_state, 0.0)
    out_fp = model(x)["one2one"]["boxes"].detach().clone()
    set_qat_ratio(qat_state, 1.0)
    out_q = model(x)["one2one"]["boxes"].detach().clone()
    diff = (out_fp - out_q).abs().mean().item()
    assert diff > 0.0, "ratio=0 (float) and ratio=1 (fully quantized) should not be numerically identical"
    print(f"[OK] QAT hooks: {len(patched)} layers patched, grad flows, fp/quant output diff={diff:.4f}")


def check_card_controller() -> None:
    model = build_model()
    model.train()
    x = torch.randn(2, 3, 256, 256)

    parser = argparse.ArgumentParser()
    add_card_args(parser)
    args = parser.parse_args(["--card_enable", "--card_prune_target", "0.3", "--card_prune_interval", "2"])

    ctrl = CardController(model, args)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for step in range(5):
        out = model(x)
        loss = out["one2one"]["boxes"].float().sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        stats = ctrl.step(global_step=step, steps_per_epoch=10, avg_conf=0.8)
        assert set(stats.keys()) == {"card_prune_target", "card_sparsity", "card_quant_ratio"}
    print("[OK] CardController: 5 training steps with pruning+QAT active, no shape/dtype errors")


if __name__ == "__main__":
    check_forward_shapes()
    check_card_schedule()
    check_pruning()
    check_qat()
    check_card_controller()
    print("\nALL RT-SFOD-LITE SMOKE TESTS PASSED")
