"""Physically compact a trained RASP Student and save an Ultralytics checkpoint.

The Stage-2 latent checkpoint is still the original dense YOLO26 shape.  This
script reads the accepted RASP masks and slices each Bottleneck cv1 hidden output
(+BN) and matching cv2 hidden input.  External block widths are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO  # noqa: E402
from rasp_pruning import export_compact_model, pruner_from_state, recursive_max_abs_diff  # noqa: E402
from stage2_rtsfod_yolo26 import resolve_device  # noqa: E402


def nparams(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main(args):
    device = resolve_device(args.device)
    state = torch.load(args.state, map_location=device, weights_only=False)
    rasp_state = state.get("rasp", state)
    if not rasp_state.get("enabled", True):
        raise RuntimeError("The supplied state has no enabled RASP masks.")

    base_ckpt = args.latent_model or args.stage1_model
    if not base_ckpt:
        raise ValueError("Provide --latent_model or --stage1_model")
    wrapper = YOLO(base_ckpt)
    model = wrapper.model.to(device).float()
    if "student_state" in state:
        model.load_state_dict(state["student_state"], strict=True)

    dense_params = nparams(model)
    compact = export_compact_model(model, rasp_state)
    compact = compact.to(device).float().eval()
    compact_params = nparams(compact)

    max_diff = None
    if args.verify:
        masked = copy.deepcopy(model).to(device).float().eval()
        p = pruner_from_state(masked, rasp_state["pruner"])
        p.load_state_dict(rasp_state["pruner"], strict=True)
        p.install()
        x = torch.randn(1, 3, args.verify_imgsz, args.verify_imgsz, device=device)
        with torch.no_grad():
            y_mask = masked(x)
            y_compact = compact(x)
        max_diff = recursive_max_abs_diff(y_mask, y_compact)
        p.uninstall()
        print(f"masked-vs-compact max_abs_diff={max_diff:.6g}")
        if not torch.isfinite(torch.tensor(max_diff)) or max_diff > args.verify_tol:
            raise RuntimeError(
                f"Physical compaction verification failed: diff={max_diff} > tol={args.verify_tol}. "
                "Do not report compact metrics until this is resolved."
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper.model = compact
    wrapper.save(str(out))

    report = {
        "dense_params": dense_params,
        "compact_params": compact_params,
        "parameter_reduction_fraction": 1.0 - compact_params / max(dense_params, 1),
        "masked_vs_compact_max_abs_diff": max_diff,
        "source_state": str(args.state),
        "output": str(out),
    }

    # Optional actual MAC count using the same DepGraph ecosystem dependency.
    if args.count_macs:
        try:
            import torch_pruning as tp

            x = torch.randn(1, 3, args.verify_imgsz, args.verify_imgsz, device=device)
            dense_for_count = model.eval()
            dense_macs, _ = tp.utils.count_ops_and_params(dense_for_count, x)
            compact_macs, _ = tp.utils.count_ops_and_params(compact, x)
            report.update(
                {
                    "dense_macs": float(dense_macs),
                    "compact_macs": float(compact_macs),
                    "mac_reduction_fraction": 1.0 - float(compact_macs) / max(float(dense_macs), 1.0),
                    "mac_imgsz": args.verify_imgsz,
                }
            )
        except Exception as exc:
            report["mac_count_error"] = f"{type(exc).__name__}: {exc}"

    report_path = out.with_suffix(out.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"dense_params={dense_params/1e6:.3f}M compact_params={compact_params/1e6:.3f}M ")
    print(f"parameter_reduction={report['parameter_reduction_fraction']:.2%}")
    print(f"saved compact model: {out}")
    print(f"saved report: {report_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True, help="rasp_training_state_epoch_*.pt")
    p.add_argument("--latent_model", default="", help="Dense latent student checkpoint saved by Stage 2")
    p.add_argument("--stage1_model", default="", help="Fallback architecture checkpoint; student_state is loaded from --state")
    p.add_argument("--out", required=True, help="Output compact Ultralytics .pt")
    p.add_argument("--device", default="0")
    p.add_argument("--verify", action="store_true", help="Compare masked and physically compact predictions")
    p.add_argument("--verify_imgsz", type=int, default=256)
    p.add_argument("--verify_tol", type=float, default=2e-4)
    p.add_argument("--count_macs", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
