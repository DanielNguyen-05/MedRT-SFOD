"""CLI for architecture_benchmark.py — Table 1-style params/FPS/latency comparison
across the real proposed architectures (yolo26 baseline, yolo26-lite, yolo26-lite-seg).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from scripts.legacy.architecture_benchmark import ARCHITECTURES, run_benchmark  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="RT-SFOD-Lite architecture benchmark (real model configs)")
    parser.add_argument(
        "--arch",
        type=str,
        choices=list(ARCHITECTURES) + ["all"],
        default="all",
        help="Which architecture to benchmark",
    )
    parser.add_argument("--scale", type=str, default="n", choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    names = list(ARCHITECTURES) if args.arch == "all" else [args.arch]

    print("=" * 88)
    print("RT-SFOD-Lite Architecture Benchmark")
    print("=" * 88)
    print(f"{'Architecture':<24}{'Task':<10}{'Scale':<7}{'Params (M)':<12}{'FPS':<10}{'Latency (ms)':<14}")
    print("-" * 88)

    results = []
    for name in names:
        result = run_benchmark(name, device=device, scale=args.scale)
        results.append(result)
        print(
            f"{result['architecture']:<24}{result['task']:<10}{result['scale']:<7}"
            f"{result['params_m']:<12}{result['fps']:<10}{result['latency_ms']:<14}"
        )
    print("=" * 88)
    print("NOTE: FPS/latency measured on:", device, "— paper-comparable numbers need the same GPU as Table 1 (RTX A6000, FP32 PyTorch).")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output_json}")


if __name__ == "__main__":
    main()
