#!/usr/bin/env python3
"""
Reproducible MedRT-SFSeg latency/FPS benchmark.

Reports two deployment-oriented measurements at batch=1:
1) network_forward: preprocessed tensor -> model forward
2) end_to_end: in-memory BGR image -> Ultralytics preprocess -> forward -> postprocess

Disk I/O is excluded from end-to-end timing because the image is loaded once before
benchmarking. CUDA synchronization is used around timed regions.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def resolve_device(s: str) -> torch.device:
    if s.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if s.isdigit():
        return torch.device(f"cuda:{s}")
    return torch.device(s)


def stats_ms(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    return {
        "mean_ms": mean,
        "std_ms": float(x.std()),
        "median_ms": float(np.median(x)),
        "p90_ms": float(np.percentile(x, 90)),
        "p95_ms": float(np.percentile(x, 95)),
        "min_ms": float(x.min()),
        "max_ms": float(x.max()),
        "fps_from_mean": float(1000.0 / mean),
    }


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_network_forward(
    core: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            _ = core(x)
        sync(device)

        times: list[float] = []
        if device.type == "cuda":
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = core(x)
                end.record()
                end.synchronize()
                times.append(float(start.elapsed_time(end)))
        else:
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = core(x)
                times.append((time.perf_counter() - t0) * 1000.0)
    return times


def time_end_to_end(
    model: YOLO,
    image_bgr: np.ndarray,
    device_arg: str,
    imgsz: int,
    conf: float,
    precision: str,
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[float]:
    half = precision == "fp16" and device.type == "cuda"

    for _ in range(warmup):
        _ = model.predict(
            source=image_bgr,
            imgsz=imgsz,
            conf=conf,
            device=device_arg,
            half=half,
            verbose=False,
        )
    sync(device)

    times: list[float] = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        _ = model.predict(
            source=image_bgr,
            imgsz=imgsz,
            conf=conf,
            device=device_arg,
            half=half,
            verbose=False,
        )
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--image", required=True, help="Representative image; loaded once, disk I/O excluded")
    ap.add_argument("--out", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument(
        "--no-fuse",
        action="store_true",
        help="Do not fuse the raw network before network-forward benchmark.",
    )
    args = ap.parse_args()

    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    out_path = Path(args.out).resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    device = resolve_device(args.device)
    if args.precision == "fp16" and device.type != "cuda":
        raise ValueError("FP16 benchmark requires CUDA")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    print("=" * 78)
    print("MEDRT-SFSEG SPEED BENCHMARK")
    print("=" * 78)
    print("model     :", model_path)
    print("GPU/device:", device)
    if device.type == "cuda":
        print("GPU name  :", torch.cuda.get_device_name(device))
    print("input     :", f"1x3x{args.imgsz}x{args.imgsz}")
    print("precision :", args.precision)
    print("warmup    :", args.warmup)
    print("iterations:", args.iters)
    print()

    # Raw network benchmark: separate wrapper so dtype/fusion cannot affect E2E wrapper.
    raw_wrapper = YOLO(str(model_path), task="segment")
    full_graph_params = sum(p.numel() for p in raw_wrapper.model.parameters())
    core = raw_wrapper.model.to(device).eval()
    if not args.no_fuse and hasattr(core, "fuse"):
        fused = core.fuse()
        if fused is not None:
            core = fused
        core = core.to(device).eval()

    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    if dtype == torch.float16:
        core = core.half()
    else:
        core = core.float()

    deployment_params = sum(p.numel() for p in core.parameters())

    x = torch.rand(
        (1, 3, args.imgsz, args.imgsz),
        device=device,
        dtype=dtype,
    )

    network_times = time_network_forward(
        core=core,
        x=x,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    network_stats = stats_ms(network_times)

    # End-to-end benchmark: independent wrapper, in-memory image, no disk I/O.
    e2e_wrapper = YOLO(str(model_path), task="segment")
    e2e_times = time_end_to_end(
        model=e2e_wrapper,
        image_bgr=image_bgr,
        device_arg=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        precision=args.precision,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    e2e_stats = stats_ms(e2e_times)

    file_size_mb = model_path.stat().st_size / (1024.0 ** 2)

    payload = {
        "model": str(model_path),
        "representative_image": str(image_path),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "imgsz": args.imgsz,
        "batch": 1,
        "precision": args.precision,
        "warmup": args.warmup,
        "iterations": args.iters,
        "conf": args.conf,
        "disk_io_included": False,
        "raw_network_fused": not args.no_fuse,
        "full_training_graph_params": int(full_graph_params),
        "full_training_graph_params_M": float(full_graph_params / 1e6),
        "deployment_fused_params": int(deployment_params),
        "deployment_fused_params_M": float(deployment_params / 1e6),
        "checkpoint_size_MB": float(file_size_mb),
        "network_forward": network_stats,
        "end_to_end": e2e_stats,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("NETWORK FORWARD")
    print(f"  mean latency : {network_stats['mean_ms']:.3f} ms/image")
    print(f"  median       : {network_stats['median_ms']:.3f} ms/image")
    print(f"  p95          : {network_stats['p95_ms']:.3f} ms/image")
    print(f"  FPS          : {network_stats['fps_from_mean']:.2f}")
    print()
    print("END-TO-END (in-memory image, disk I/O excluded)")
    print(f"  mean latency : {e2e_stats['mean_ms']:.3f} ms/image")
    print(f"  median       : {e2e_stats['median_ms']:.3f} ms/image")
    print(f"  p95          : {e2e_stats['p95_ms']:.3f} ms/image")
    print(f"  FPS          : {e2e_stats['fps_from_mean']:.2f}")
    print()
    print(f"Full graph params : {full_graph_params / 1e6:.3f} M")
    print(f"Deployment params : {deployment_params / 1e6:.3f} M")
    print(f"Checkpoint size: {file_size_mb:.2f} MB")
    print("Saved          :", out_path)
    print("[PASS] Speed benchmark completed")


if __name__ == "__main__":
    main()
