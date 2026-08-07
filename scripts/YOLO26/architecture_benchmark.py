"""Paper-safe architecture benchmark for real YOLO26 configs.

Fixes a subtle bug in the earlier benchmark: ``--scale`` was printed but never
applied to the YAML, so n/s/m/l/x rows could silently benchmark the same model.
This revision edits the parsed YAML's ``scale`` field before model construction.

It also uses CUDA Events on GPU and allows HxW to be specified explicitly.  Do
not call mask-based training sparsity a speedup: only physically exported models
should be used for compressed latency/FPS claims.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics.nn.tasks import DetectionModel, SegmentationModel  # noqa: E402

MODEL_ROOT = REPO_ROOT / "ultralytics" / "cfg" / "models" / "26"
ARCHITECTURES: Dict[str, tuple[str, str, int, tuple[int, int]]] = {
    "yolo26_baseline": ("yolo26.yaml", "detect", 8, (1024, 1024)),
    "yolo26_lite": ("yolo26-lite.yaml", "detect", 8, (1024, 1024)),
    "yolo26_lite_seg_polyp": ("yolo26-lite-seg.yaml", "segment", 1, (640, 640)),
}


def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def build_model(yaml_name: str, task: str, nc: int, scale: str = "n") -> nn.Module:
    cfg_path = MODEL_ROOT / yaml_name
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["scale"] = scale  # IMPORTANT: actually apply the requested scale.
    cls = SegmentationModel if task == "segment" else DetectionModel
    model = cls(cfg=cfg, ch=3, nc=nc, verbose=False)
    model.eval()
    return model


@torch.inference_mode()
def benchmark_inference(
    model: nn.Module,
    height: int,
    width: int,
    batch_size: int,
    num_warmup: int,
    num_runs: int,
    device: torch.device,
) -> Dict[str, float]:
    model = model.to(device).eval()
    x = torch.randn(batch_size, 3, height, width, device=device)

    for _ in range(num_warmup):
        _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(num_runs):
            _ = model(x)
        end.record()
        torch.cuda.synchronize(device)
        total_ms = float(start.elapsed_time(end))
        latency_ms = total_ms / num_runs
        elapsed_s = total_ms / 1000.0
    else:
        t0 = time.perf_counter()
        for _ in range(num_runs):
            _ = model(x)
        elapsed_s = time.perf_counter() - t0
        latency_ms = elapsed_s * 1000.0 / num_runs

    fps = num_runs * batch_size / max(elapsed_s, 1e-12)
    return {"fps": fps, "latency_ms": latency_ms}


def run_benchmark(
    name: str,
    device: torch.device,
    scale: str = "n",
    height: int | None = None,
    width: int | None = None,
    batch_size: int = 1,
    num_warmup: int = 20,
    num_runs: int = 100,
) -> dict:
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {name!r}; choices={list(ARCHITECTURES)}")
    yaml_name, task, nc, default_hw = ARCHITECTURES[name]
    h = int(height or default_hw[0])
    w = int(width or default_hw[1])
    model = build_model(yaml_name, task, nc, scale)
    metrics = benchmark_inference(model, h, w, batch_size, num_warmup, num_runs, device)
    return {
        "architecture": name,
        "yaml": yaml_name,
        "task": task,
        "scale": scale,
        "input_hw": [h, w],
        "batch_size": batch_size,
        "params_m": round(count_parameters(model), 4),
        "fps": round(metrics["fps"], 2),
        "latency_ms": round(metrics["latency_ms"], 3),
        "device": str(device),
        "precision": "FP32",
        "runtime": "PyTorch",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=list(ARCHITECTURES) + ["all"], default="all")
    ap.add_argument("--scale", choices=["n", "s", "m", "l", "x"], default="n")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    names = list(ARCHITECTURES) if args.arch == "all" else [args.arch]
    results = [
        run_benchmark(n, device, args.scale, args.height, args.width, args.batch, args.warmup, args.runs)
        for n in names
    ]
    print(json.dumps(results, indent=2))
    print("NOTE: compare methods only under identical hardware, precision, runtime, batch, and HxW.")
    print("NOTE: training-time channel masks are NOT a physically pruned model and must not be reported as pruning FPS gains.")
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
