"""Inspect RASP-safe Bottleneck hidden groups before expensive training.

Use this as the first go/no-go check. It reports:
- number of eligible Bottleneck hidden spaces;
- hidden channels and approximate parameters/MACs controlled by RASP;
- optional Torch-Pruning DepGraph locality audit;
- rough savings for fixed hidden sparsity levels (analysis only, not the main method).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO  # noqa: E402
from rasp_pruning import AdaptiveHiddenPruner, save_audit_json  # noqa: E402
from stage2_rtsfod_yolo26 import resolve_device  # noqa: E402


def main(args):
    device = resolve_device(args.device)
    wrapper = YOLO(args.model)
    model = wrapper.model.to(device).float().eval()
    pruner = AdaptiveHiddenPruner(
        model,
        min_hidden=args.min_hidden,
        min_keep_ratio=args.min_keep_ratio,
        round_to=args.round_to,
        seed=args.seed,
    )
    if not pruner.groups:
        raise RuntimeError("No RASP-safe Bottleneck hidden groups were found.")
    pruner.install()

    x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
    pruner.collect_importance = False
    with torch.no_grad():
        _ = model(x)
    pruner.collect_importance = True

    total_macs, total_params = pruner.total_prunable_cost()
    model_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}")
    print(f"total_model_params={model_params/1e6:.3f}M")
    print(f"eligible_hidden_groups={len(pruner.groups)}")
    print(f"eligible_hidden_channels={sum(g.hidden for g in pruner.groups.values())}")
    print(f"RASP-controlled parameter terms≈{total_params/1e6:.3f}M")
    print(f"RASP-controlled MAC terms@{args.imgsz}≈{total_macs/1e9:.3f}G")

    rows = []
    for name, item in pruner.groups.items():
        macs, params = pruner.per_channel_cost(item)
        rows.append(
            {
                "name": name,
                "hidden": item.hidden,
                "hw": item.output_hw,
                "mac_per_hidden": macs,
                "params_per_hidden": params,
            }
        )
    rows.sort(key=lambda r: r["mac_per_hidden"] * r["hidden"], reverse=True)
    print("\nTop groups by controlled MACs:")
    for r in rows[: args.topk]:
        print(
            f"  {r['name']:<55} hidden={r['hidden']:4d} hw={r['hw']} "
            f"MAC≈{r['mac_per_hidden']*r['hidden']/1e9:7.3f}G "
            f"params≈{r['params_per_hidden']*r['hidden']/1e6:6.3f}M"
        )

    print("\nStatic sparsity estimates (ablation reference only):")
    for s in (0.2, 0.3, 0.4, 0.5):
        print(
            f"  hidden_s={s:.0%}: controlled_params_saved≈{total_params*s/1e6:.3f}M, "
            f"controlled_MAC_saved≈{total_macs*s/1e9:.3f}G"
        )

    report = {
        "model": args.model,
        "imgsz": args.imgsz,
        "model_params": model_params,
        "eligible_hidden_groups": len(pruner.groups),
        "eligible_hidden_channels": sum(g.hidden for g in pruner.groups.values()),
        "controlled_params": total_params,
        "controlled_macs": total_macs,
        "groups": rows,
    }

    if args.depgraph:
        print("\nRunning DepGraph audit...")
        audit = pruner.audit_depgraph(x, require_local=True)
        bad = [n for n, v in audit.items() if not v["ok"]]
        print(f"DepGraph local-safe: {len(audit)-len(bad)}/{len(audit)}")
        if bad:
            print("Failed groups:", bad[:10])
        report["depgraph"] = audit
        if args.require_depgraph and bad:
            raise SystemExit(f"DepGraph audit failed for {len(bad)} groups")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved audit report: {out}")

    pruner.uninstall()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="YOLO26 checkpoint, e.g. Stage-1 checkpoint")
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--device", default="0")
    p.add_argument("--min_hidden", type=int, default=16)
    p.add_argument("--min_keep_ratio", type=float, default=0.5)
    p.add_argument("--round_to", type=int, default=8)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--depgraph", action="store_true")
    p.add_argument("--require_depgraph", action="store_true")
    p.add_argument("--out", default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
