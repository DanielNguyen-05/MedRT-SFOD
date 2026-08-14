#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "YOLO26"))

from ultralytics import YOLO  # noqa: E402
from rasp_pruning import AdaptiveHiddenPruner  # noqa: E402


SEGMENT_EXCLUDES = (
    "one2one",
    "one2many",
    ".dfl",
    "c2psa",
    ".attn",
    ".psa",

    # Segmentation-specific hard exclusions
    "proto",
    "mask_head",
    "mask_coefficient",
)


def resolve_device(x: str) -> torch.device:
    if x.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if x.isdigit():
        return torch.device(f"cuda:{x}")
    return torch.device(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=640)

    ap.add_argument(
        "--audit-imgsz",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--min-hidden",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--min-keep-ratio",
        type=float,
        default=0.50,
    )

    ap.add_argument(
        "--round-to",
        type=int,
        default=8,
    )

    ap.add_argument(
        "--depgraph",
        action="store_true",
    )

    ap.add_argument(
        "--out",
        default=(
            "runs/seg/rasp_sfseg/"
            "rasp_seg_structural_audit.json"
        ),
    )

    args = ap.parse_args()

    device = resolve_device(args.device)

    ckpt = Path(args.model).resolve()

    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    wrapper = YOLO(str(ckpt))
    model = wrapper.model.to(device).float()
    model.eval()

    head = model.model[-1]

    print("=" * 78)
    print("RASP-SFSEG STRUCTURAL AUDIT")
    print("=" * 78)

    print("model       :", ckpt)
    print("model class :", type(model).__name__)
    print("head class  :", type(head).__name__)
    print("end2end     :", getattr(model, "end2end", None))
    print("nc          :", getattr(head, "nc", None))
    print("nm          :", getattr(head, "nm", None))
    print("npr         :", getattr(head, "npr", None))
    print()

    if "segment" not in type(head).__name__.lower():
        raise RuntimeError(
            f"Expected segmentation head, got {type(head).__name__}"
        )

    if not getattr(model, "end2end", False):
        raise RuntimeError(
            "Expected end-to-end dual-head segmentation model"
        )

    pruner = AdaptiveHiddenPruner(
        model,
        importance_beta=0.90,
        min_hidden=args.min_hidden,
        min_keep_ratio=args.min_keep_ratio,
        round_to=args.round_to,
        gmm_posterior=0.80,
        gmm_bic_gain=0.0,
        gmm_min_separation=1.0,
        gmm_min_samples=16,
        cost_gamma=1.0,
        seed=29,
        exclude_name_substrings=SEGMENT_EXCLUDES,
    )

    if not pruner.groups:
        raise RuntimeError(
            "No eligible Bottleneck hidden groups found."
        )

    # Observe actual feature-map sizes for MAC estimates.
    pruner.collect_importance = False
    pruner.install()

    dummy = torch.randn(
        1,
        3,
        args.imgsz,
        args.imgsz,
        device=device,
    )

    with torch.no_grad():
        _ = model(dummy)

    pruner.uninstall()

    total_model_params = count_params(model)

    total_hidden = sum(
        g.hidden
        for g in pruner.groups.values()
    )

    total_prunable_macs, total_prunable_params = (
        pruner.total_prunable_cost()
    )

    max_remove_channels = 0
    max_remove_macs = 0.0
    max_remove_params = 0.0

    group_rows = []

    print(
        f"eligible groups   : {len(pruner.groups)}"
    )

    print(
        f"eligible channels : {total_hidden}"
    )

    print()

    print(
        f"{'group':52s} "
        f"{'H':>6s} "
        f"{'min_keep':>8s} "
        f"{'max_rm':>7s} "
        f"{'HxW':>12s}"
    )

    print("-" * 92)

    for name, g in pruner.groups.items():

        min_keep = max(
            args.min_hidden,
            int(
                __import__("math").ceil(
                    g.hidden
                    * args.min_keep_ratio
                )
            ),
        )

        # Keep hardware-friendly final width.
        max_rm = max(
            0,
            g.hidden - min_keep,
        )

        max_rm = (
            max_rm // args.round_to
        ) * args.round_to

        c_macs, c_params = (
            pruner.per_channel_cost(g)
        )

        max_remove_channels += max_rm
        max_remove_macs += (
            c_macs * max_rm
        )
        max_remove_params += (
            c_params * max_rm
        )

        hw = (
            f"{g.output_hw[0]}x{g.output_hw[1]}"
            if g.output_hw is not None
            else "?"
        )

        print(
            f"{name:52s} "
            f"{g.hidden:6d} "
            f"{min_keep:8d} "
            f"{max_rm:7d} "
            f"{hw:>12s}"
        )

        low = name.lower()

        if (
            "proto" in low
            or "one2one" in low
            or "one2many" in low
            or "mask" in low
        ):
            raise RuntimeError(
                "Unsafe segmentation-head group discovered: "
                + name
            )

        group_rows.append(
            {
                "name": name,
                "hidden": g.hidden,
                "min_keep": min_keep,
                "max_remove": max_rm,
                "output_hw": g.output_hw,
                "per_channel_macs": c_macs,
                "per_channel_params": c_params,
            }
        )

    print()
    print(
        "model params            :",
        f"{total_model_params / 1e6:.3f} M",
    )

    print(
        "safe hidden params      :",
        f"{total_prunable_params / 1e6:.3f} M",
    )

    print(
        "safe hidden MACs@audit  :",
        f"{total_prunable_macs / 1e9:.3f} G",
    )

    print(
        "max removable channels  :",
        max_remove_channels,
    )

    print(
        "max removable params    :",
        f"{max_remove_params / 1e6:.3f} M",
    )

    print(
        "max removable MACs      :",
        f"{max_remove_macs / 1e9:.3f} G",
    )

    print(
        "max model-param fraction:",
        f"{max_remove_params / max(total_model_params, 1):.4f}",
    )

    depgraph = {}

    if args.depgraph:
        print()
        print(
            "[RASP] running local DepGraph audit..."
        )

        example = torch.randn(
            1,
            3,
            args.audit_imgsz,
            args.audit_imgsz,
            device=device,
        )

        depgraph = pruner.audit_depgraph(
            example,
            require_local=True,
        )

        bad = [
            name
            for name, row
            in depgraph.items()
            if not row["ok"]
        ]

        print(
            "DepGraph local-safe : "
            f"{len(depgraph)-len(bad)}/"
            f"{len(depgraph)}"
        )

        if bad:
            print(
                "UNSAFE GROUPS:",
                bad,
            )

            raise RuntimeError(
                f"{len(bad)} groups failed "
                "DepGraph locality audit"
            )

    result = {
        "model": str(ckpt),
        "model_class": type(model).__name__,
        "head_class": type(head).__name__,
        "end2end": bool(
            getattr(model, "end2end", False)
        ),
        "nc": getattr(head, "nc", None),
        "nm": getattr(head, "nm", None),
        "npr": getattr(head, "npr", None),

        "segment_head_pruned": False,
        "proto_pruned": False,

        "eligible_groups":
            len(pruner.groups),

        "eligible_channels":
            total_hidden,

        "model_params":
            total_model_params,

        "safe_hidden_params":
            total_prunable_params,

        "safe_hidden_macs":
            total_prunable_macs,

        "max_remove_channels":
            max_remove_channels,

        "max_remove_params":
            max_remove_params,

        "max_remove_macs":
            max_remove_macs,

        "groups":
            group_rows,

        "depgraph":
            depgraph,
    }

    out = Path(args.out).resolve()

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("[PASS] RASP-SFSeg structural audit")
    print("saved:", out)


if __name__ == "__main__":
    main()
