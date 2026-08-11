"""Physically compact a trained RASP Student and save an Ultralytics checkpoint.

The Stage-2 latent checkpoint is still the original dense YOLO26 shape.
This script reads the accepted RASP masks and slices each Bottleneck cv1 hidden
output (+BN) and matching cv2 hidden input. External block widths are unchanged.

Important:
    Physical equivalence is verified on CPU.

    The masked dense model computes the original full hidden convolution and then
    applies zero gates, whereas the compact model performs a physically smaller
    convolution. These operations are algebraically equivalent, but CUDA/cuDNN
    may select different kernels/reduction orders after the tensor shapes change,
    resulting in larger floating-point drift.

    CPU verification therefore provides the deterministic structural-equivalence
    check used before compact Params/MACs are reported.
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
from rasp_pruning import (  # noqa: E402
    export_compact_model,
    pruner_from_state,
    recursive_max_abs_diff,
)
from stage2_rtsfod_yolo26 import resolve_device  # noqa: E402


def nparams(model) -> int:
    return sum(p.numel() for p in model.parameters())


def split_eval_output(output):
    """Split YOLO eval output into postprocessed output and raw prediction dict."""
    if (
        isinstance(output, tuple)
        and len(output) == 2
        and isinstance(output[1], dict)
    ):
        return output[0], output[1]

    return output, None


def main(args):
    device = resolve_device(args.device)

    # ------------------------------------------------------------------
    # Load RASP state
    # ------------------------------------------------------------------
    state = torch.load(
        args.state,
        map_location=device,
        weights_only=False,
    )

    rasp_state = state.get("rasp", state)

    if not rasp_state.get("enabled", True):
        raise RuntimeError(
            "The supplied state has no enabled RASP masks."
        )

    # ------------------------------------------------------------------
    # Reconstruct exact dense latent Student
    # ------------------------------------------------------------------
    base_ckpt = args.latent_model or args.stage1_model

    if not base_ckpt:
        raise ValueError(
            "Provide --latent_model or --stage1_model"
        )

    wrapper = YOLO(base_ckpt)

    model = wrapper.model.to(device).float()

    # Prefer exact Student parameters stored in final training state.
    if "student_state" in state:
        model.load_state_dict(
            state["student_state"],
            strict=True,
        )

    model.eval()

    dense_params = nparams(model)

    # ------------------------------------------------------------------
    # Physical compaction
    # ------------------------------------------------------------------
    compact = export_compact_model(
        model,
        rasp_state,
    )

    compact = compact.to(device).float().eval()

    compact_params = nparams(compact)

    # ------------------------------------------------------------------
    # Numerical equivalence verification
    # ------------------------------------------------------------------
    max_diff = None
    raw_diff = None
    post_diff = None
    verification_device = None

    if args.verify:
        # --------------------------------------------------------------
        # IMPORTANT:
        # Verify physical equivalence on CPU, not CUDA.
        #
        # Dense masked:
        #   full convolution -> gate zero channels
        #
        # Compact:
        #   smaller physical convolution
        #
        # Algebraically equivalent, but CUDA/cuDNN may use different
        # reduction algorithms for the changed convolution shapes.
        # --------------------------------------------------------------
        verify_device = torch.device("cpu")
        verification_device = "cpu"

        print("=" * 72)
        print("PHYSICAL COMPACTION VERIFICATION")
        print("=" * 72)
        print("verification_device=cpu")
        print(f"verification_imgsz={args.verify_imgsz}")
        print(f"verification_tol={args.verify_tol:.6g}")

        # --------------------------------------------------------------
        # A) Masked dense Student
        # --------------------------------------------------------------
        masked = copy.deepcopy(model)
        masked = masked.to(verify_device).float().eval()

        pruner = pruner_from_state(
            masked,
            rasp_state["pruner"],
        )

        pruner.load_state_dict(
            rasp_state["pruner"],
            strict=True,
        )

        pruner.install()

        # --------------------------------------------------------------
        # B) Physical compact Student
        # --------------------------------------------------------------
        compact_verify = copy.deepcopy(compact)
        compact_verify = compact_verify.to(
            verify_device
        ).float().eval()

        # --------------------------------------------------------------
        # Deterministic common input
        # --------------------------------------------------------------
        torch.manual_seed(12345)

        x_verify = torch.randn(
            1,
            3,
            args.verify_imgsz,
            args.verify_imgsz,
            device=verify_device,
            dtype=torch.float32,
        )

        try:
            with torch.no_grad():
                y_mask = masked(x_verify)
                y_compact = compact_verify(x_verify)

            # Full nested YOLO output.
            max_diff = recursive_max_abs_diff(
                y_mask,
                y_compact,
            )

            # Also report raw/postprocessed differences independently.
            post_mask, raw_mask = split_eval_output(
                y_mask
            )
            post_compact, raw_compact = split_eval_output(
                y_compact
            )

            post_diff = recursive_max_abs_diff(
                post_mask,
                post_compact,
            )

            if (
                raw_mask is not None
                and raw_compact is not None
            ):
                raw_diff = recursive_max_abs_diff(
                    raw_mask,
                    raw_compact,
                )

            print()
            print(
                f"masked-vs-compact CPU full max_abs_diff="
                f"{max_diff:.9g}"
            )

            print(
                f"masked-vs-compact CPU postprocess max_abs_diff="
                f"{post_diff:.9g}"
            )

            if raw_diff is not None:
                print(
                    f"masked-vs-compact CPU raw max_abs_diff="
                    f"{raw_diff:.9g}"
                )

            # ----------------------------------------------------------
            # Full CPU output is the required equivalence criterion.
            # ----------------------------------------------------------
            finite = torch.isfinite(
                torch.tensor(max_diff)
            ).item()

            if not finite or max_diff > args.verify_tol:
                raise RuntimeError(
                    "Physical compaction verification failed on CPU: "
                    f"diff={max_diff} > tol={args.verify_tol}. "
                    "Do not report compact metrics until this is resolved."
                )

            print()
            print(
                "[PASS] Physical compaction numerical equivalence"
            )

        finally:
            pruner.uninstall()

    # ------------------------------------------------------------------
    # Save physical compact checkpoint
    # ------------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wrapper.model = compact
    wrapper.save(str(out))

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report = {
        "dense_params": dense_params,
        "compact_params": compact_params,
        "parameter_reduction_fraction":
            1.0 - compact_params / max(dense_params, 1),

        "verification_enabled": bool(args.verify),
        "verification_device": verification_device,
        "verification_imgsz":
            args.verify_imgsz if args.verify else None,
        "verification_tolerance":
            args.verify_tol if args.verify else None,

        "masked_vs_compact_max_abs_diff":
            max_diff,
        "masked_vs_compact_raw_max_abs_diff":
            raw_diff,
        "masked_vs_compact_postprocess_max_abs_diff":
            post_diff,

        "source_state": str(args.state),
        "latent_model": str(base_ckpt),
        "output": str(out),
    }

    # ------------------------------------------------------------------
    # Actual MAC count
    #
    # MAC counting can still use the requested GPU/device. It is separate
    # from the CPU numerical-equivalence verification.
    # ------------------------------------------------------------------
    if args.count_macs:
        try:
            import torch_pruning as tp

            x_macs = torch.randn(
                1,
                3,
                args.verify_imgsz,
                args.verify_imgsz,
                device=device,
            )

            dense_for_count = model.to(
                device
            ).float().eval()

            compact_for_count = compact.to(
                device
            ).float().eval()

            dense_macs, dense_count_params = (
                tp.utils.count_ops_and_params(
                    dense_for_count,
                    x_macs,
                )
            )

            compact_macs, compact_count_params = (
                tp.utils.count_ops_and_params(
                    compact_for_count,
                    x_macs,
                )
            )

            dense_macs = float(dense_macs)
            compact_macs = float(compact_macs)

            report.update(
                {
                    "dense_macs": dense_macs,
                    "compact_macs": compact_macs,

                    "mac_reduction_fraction":
                        1.0
                        - compact_macs
                        / max(dense_macs, 1.0),

                    "mac_imgsz":
                        args.verify_imgsz,

                    "mac_count_device":
                        str(device),

                    "dense_count_params":
                        int(dense_count_params),

                    "compact_count_params":
                        int(compact_count_params),
                }
            )

        except Exception as exc:
            report["mac_count_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    report_path = out.with_suffix(
        out.suffix + ".report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("RASP PHYSICAL COMPACT EXPORT")
    print("=" * 72)

    print(
        f"dense_params   = "
        f"{dense_params / 1e6:.3f}M"
    )

    print(
        f"compact_params = "
        f"{compact_params / 1e6:.3f}M"
    )

    print(
        f"parameter_reduction = "
        f"{report['parameter_reduction_fraction']:.2%}"
    )

    if "dense_macs" in report:
        print(
            f"dense_MACs@{args.verify_imgsz} = "
            f"{report['dense_macs'] / 1e9:.3f}G"
        )

        print(
            f"compact_MACs@{args.verify_imgsz} = "
            f"{report['compact_macs'] / 1e9:.3f}G"
        )

        print(
            f"MAC_reduction = "
            f"{report['mac_reduction_fraction']:.2%}"
        )

    print()
    print(f"saved compact model: {out}")
    print(f"saved report: {report_path}")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--state",
        required=True,
        help="rasp_training_state_epoch_*.pt",
    )

    p.add_argument(
        "--latent_model",
        default="",
        help="Dense latent student checkpoint saved by Stage 2",
    )

    p.add_argument(
        "--stage1_model",
        default="",
        help=(
            "Fallback architecture checkpoint; "
            "student_state is loaded from --state"
        ),
    )

    p.add_argument(
        "--out",
        required=True,
        help="Output compact Ultralytics .pt",
    )

    p.add_argument(
        "--device",
        default="0",
    )

    p.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Verify masked-vs-compact numerical equivalence "
            "on CPU"
        ),
    )

    p.add_argument(
        "--verify_imgsz",
        type=int,
        default=256,
    )

    p.add_argument(
        "--verify_tol",
        type=float,
        default=2e-4,
    )

    p.add_argument(
        "--count_macs",
        action="store_true",
    )

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
