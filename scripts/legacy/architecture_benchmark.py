"""Architecture benchmark — params + FPS/latency for the REAL model configs.

Replaces the previous version of this file, which benchmarked
`CityDetectionModel` / `PolypDetectionModel` / `PolypSegmentationModel` — three
hand-written `nn.Sequential` CNNs unrelated to the actual proposed
architecture (`yolo26-lite.yaml` with the `C2fFaster` backbone, DHF, MARD,
CARD). Benchmarking those toy models and reporting the numbers as if they
characterized the proposed method would be a materially misleading result
for a paper (Table 1-style comparison needs to measure the model you're
actually proposing).

This version benchmarks the real model YAML configs directly via
`ultralytics.nn.tasks.DetectionModel` / `SegmentationModel`, the same
mechanism validated in `test_card_lite_smoke.py`. It does not require
`ultralytics/data/` to be present (see README) since it never touches the
dataset loader.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics.nn.tasks import DetectionModel, SegmentationModel  # noqa: E402

MODEL_ROOT = REPO_ROOT / "ultralytics" / "cfg" / "models" / "26"

# name -> (yaml filename, task, default nc, default imgsz)
ARCHITECTURES: Dict[str, tuple[str, str, int, int]] = {
    "yolo26_baseline": ("yolo26.yaml", "detect", 8, 1024),  # 8 classes: Cityscapes/C2F convention
    "yolo26_lite": ("yolo26-lite.yaml", "detect", 8, 1024),  # Try 1: C2fFaster backbone
    "yolo26_lite_seg_polyp": ("yolo26-lite-seg.yaml", "segment", 1, 640),  # Try 3: 1 class (polyp)
}


def count_parameters(model: nn.Module) -> float:
    """Total parameter count in millions (all params, not just trainable —
    matches how the RT-SFOD paper reports "Params (M)" in Table 1)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def build_model(yaml_name: str, task: str, nc: int, scale: str = "n") -> nn.Module:
    cfg_path = str(MODEL_ROOT / yaml_name)
    cls = SegmentationModel if task == "segment" else DetectionModel
    model = cls(cfg=cfg_path, ch=3, nc=nc, verbose=False)
    model.eval()
    return model


@torch.no_grad()
def benchmark_inference(
    model: nn.Module,
    imgsz: int,
    batch_size: int = 1,
    num_warmup: int = 5,
    num_runs: int = 30,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Measure FPS/latency with a dummy input, eval() mode (matches the
    RT-SFOD paper's protocol of measuring inference-time cost, not
    training-time cost — DHF/MARD/CARD are training-only by design)."""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(batch_size, 3, imgsz, imgsz, device=device)

    for _ in range(num_warmup):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(num_runs):
        _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    latency_ms = (elapsed / num_runs) * 1000
    fps = (num_runs * batch_size) / elapsed
    return {"fps": fps, "latency_ms": latency_ms}


def run_benchmark(name: str, device: torch.device, scale: str = "n") -> Dict[str, object]:
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{name}'. Choices: {list(ARCHITECTURES)}")
    yaml_name, task, nc, imgsz = ARCHITECTURES[name]
    model = build_model(yaml_name, task, nc, scale=scale)
    params_m = count_parameters(model)
    metrics = benchmark_inference(model, imgsz=imgsz, device=device)
    return {
        "architecture": name,
        "yaml": yaml_name,
        "task": task,
        "scale": scale,
        "imgsz": imgsz,
        "params_m": round(params_m, 3),
        "fps": round(metrics["fps"], 2),
        "latency_ms": round(metrics["latency_ms"], 2),
        "device": str(device),
    }
