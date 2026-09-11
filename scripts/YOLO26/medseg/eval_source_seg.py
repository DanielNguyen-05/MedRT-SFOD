#!/usr/bin/env python3
"""
Generic full evaluator for YOLO26 segmentation checkpoints.

Designed to evaluate source-supervised checkpoints and baseline YOLO26-Seg-S
checkpoints with the same protocol used for MedRT-SFSeg final evaluation.

Requires these scripts in the same directory:
  - analyze_seg_results.py
  - benchmark_seg_speed.py

Outputs:
  - one JSON file specified by --out
  - one companion Markdown table next to the JSON
  - detailed analysis/speed artifacts in <out_stem>_artifacts/

Boundary metrics:
  - ASD (mm): HEAL-style evaluator from analyze_seg_results.py
  - HD95 (mm): same unit-spacing convention as analyze_seg_results.py
  - empty prediction/GT cases are NaN for ASD/HD95 and excluded from boundary
    aggregation, while overlap metrics still include all images.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

# Force the repository-local Ultralytics package before importing it.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import torch
import ultralytics
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _float_or_none(x: Any):
    try:
        return float(x)
    except Exception:
        return None


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def _run(cmd: list[str]) -> None:
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + old if old else "")
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _metric_object(obj: Any) -> dict[str, float | None]:
    if obj is None:
        return {}
    out: dict[str, float | None] = {}
    for name in ("map50", "map", "map75", "mp", "mr"):
        out[name] = _float_or_none(getattr(obj, name, None))
    return out


def _official_val(
    model_path: Path,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    device: str,
) -> dict[str, Any]:
    wrapper = YOLO(str(model_path), task="segment")
    result = wrapper.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=0.001,
        iou=0.6,
        plots=False,
        verbose=False,
    )

    payload: dict[str, Any] = {
        "seg": _metric_object(getattr(result, "seg", None)),
        "box": _metric_object(getattr(result, "box", None)),
        "results_dict": {},
        "speed": getattr(result, "speed", None),
    }
    for k, v in (getattr(result, "results_dict", {}) or {}).items():
        fv = _float_or_none(v)
        if fv is not None:
            payload["results_dict"][str(k)] = fv
    return payload


def _deployment_profile(
    model_path: Path,
    imgsz: int,
    device_arg: str,
) -> dict[str, Any]:
    device = (
        torch.device("cpu")
        if device_arg.lower() == "cpu" or not torch.cuda.is_available()
        else torch.device(f"cuda:{device_arg}" if device_arg.isdigit() else device_arg)
    )

    wrapper = YOLO(str(model_path), task="segment")
    core = wrapper.model.to(device).float().eval()
    full_params = sum(p.numel() for p in core.parameters())

    fused = core.fuse(verbose=False) if hasattr(core, "fuse") else core
    if fused is not None:
        core = fused
    core = core.to(device).float().eval()
    deploy_params = sum(p.numel() for p in core.parameters())

    info_value = None
    try:
        info_value = core.info(verbose=False, imgsz=imgsz)
    except Exception:
        pass

    gflops = None
    if isinstance(info_value, (tuple, list)) and len(info_value) >= 4:
        gflops = _float_or_none(info_value[3])

    gmacs = None
    profile_backend = None
    profile_error = None
    try:
        import thop

        dummy = torch.zeros((1, 3, imgsz, imgsz), device=device)
        with torch.inference_mode():
            macs, _ = thop.profile(core, inputs=(dummy,), verbose=False)
        gmacs = float(macs / 1e9)
        if gflops is None:
            gflops = float(2.0 * macs / 1e9)
        profile_backend = "thop"
    except Exception as exc:
        profile_error = repr(exc)
        if gflops is not None:
            profile_backend = "ultralytics.model.info"

    return {
        "full_training_graph_params": int(full_params),
        "full_training_graph_params_M": float(full_params / 1e6),
        "deployment_fused_params": int(deploy_params),
        "deployment_fused_params_M": float(deploy_params / 1e6),
        "checkpoint_size_MB": float(model_path.stat().st_size / (1024.0**2)),
        "deployment_GMACs": gmacs,
        "deployment_GFLOPs": gflops,
        "profile_backend": profile_backend,
        "profile_error": profile_error,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt-masks", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="YOLO26-Seg-S")
    ap.add_argument("--benchmark-image", default=None)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    model_path = Path(args.model).resolve()
    data_yaml = Path(args.data).resolve()
    images = Path(args.images).resolve()
    gt_masks = Path(args.gt_masks).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    for p in (model_path, data_yaml, images, gt_masks):
        if not p.exists():
            raise FileNotFoundError(p)

    if not images.is_dir():
        raise NotADirectoryError(images)
    if not gt_masks.is_dir():
        raise NotADirectoryError(gt_masks)

    local_ultra = (REPO_ROOT / "ultralytics").resolve()
    loaded_ultra = Path(ultralytics.__file__).resolve()
    if local_ultra not in loaded_ultra.parents:
        raise RuntimeError(
            "Wrong Ultralytics package loaded. Expected local package under "
            f"{local_ultra}, got {loaded_ultra}. "
            "Run from ~/MedRT-SFOD with PYTHONPATH=$PWD."
        )

    if args.benchmark_image:
        benchmark_image = Path(args.benchmark_image).resolve()
        if not benchmark_image.exists():
            raise FileNotFoundError(benchmark_image)
    else:
        candidates = sorted(
            p for p in images.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not candidates:
            raise RuntimeError(f"No image available for benchmark in {images}")
        benchmark_image = candidates[0]

    artifact_dir = out.parent / f"{out.stem}_artifacts"
    analysis_dir = artifact_dir / "segmentation_analysis"
    speed_json = artifact_dir / f"speed_{args.precision}.json"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Official Ultralytics AP metrics.
    # ------------------------------------------------------------------
    print("\n[1/4] Official Ultralytics mask validation", flush=True)
    official = _official_val(
        model_path=model_path,
        data_yaml=data_yaml,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 2) Dice/IoU/etc + ASD/HD95 in mm.
    # Reuse the SAME analyzer as proposed models for apples-to-apples metrics.
    # ------------------------------------------------------------------
    print("\n[2/4] Detailed medical segmentation analysis", flush=True)
    analyze_script = SCRIPT_DIR / "analyze_seg_results.py"
    if not analyze_script.exists():
        raise FileNotFoundError(
            f"Missing {analyze_script}. Use the HEAL-style mm version of "
            "analyze_seg_results.py before running this evaluator."
        )

    _run([
        sys.executable,
        str(analyze_script),
        "--model", str(model_path),
        "--images", str(images),
        "--gt-masks", str(gt_masks),
        "--out-dir", str(analysis_dir),
        "--imgsz", str(args.imgsz),
        "--device", args.device,
        "--conf", str(args.conf),
        "--topk", str(args.topk),
    ])

    analysis_path = analysis_dir / "summary.json"
    if not analysis_path.exists():
        raise FileNotFoundError(analysis_path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # 3) Same deployment speed benchmark as proposed models.
    # ------------------------------------------------------------------
    print("\n[3/4] Deployment speed benchmark", flush=True)
    speed_script = SCRIPT_DIR / "benchmark_seg_speed.py"
    if not speed_script.exists():
        raise FileNotFoundError(speed_script)

    _run([
        sys.executable,
        str(speed_script),
        "--model", str(model_path),
        "--image", str(benchmark_image),
        "--out", str(speed_json),
        "--imgsz", str(args.imgsz),
        "--device", args.device,
        "--conf", str(args.conf),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--precision", args.precision,
    ])
    speed = json.loads(speed_json.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # 4) Params / GMACs / GFLOPs / checkpoint size.
    # ------------------------------------------------------------------
    print("\n[4/4] Deployment graph profile", flush=True)
    profile = _deployment_profile(
        model_path=model_path,
        imgsz=args.imgsz,
        device_arg=args.device,
    )

    seg = official.get("seg", {})
    macro = analysis.get("medical_macro", {})
    distributions = analysis.get("distributions", {})
    asd_dist = distributions.get("asd", {})
    hd95_dist = distributions.get("hd95", {})
    e2e = speed.get("end_to_end", {})
    net = speed.get("network_forward", {})

    mean_e2e = _float_or_none(e2e.get("mean_ms"))
    e2e_fps = _float_or_none(e2e.get("fps_from_mean"))
    real_time_pass = bool(
        mean_e2e is not None
        and e2e_fps is not None
        and mean_e2e <= 33.333333
        and e2e_fps >= 30.0
    )

    payload = {
        "method": args.method,
        "model": str(model_path),
        "num_images": analysis.get("num_images"),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "evaluation_only_gt": True,
        "official_mask_metrics": {
            "mAP50": seg.get("map50"),
            "mAP50_95": seg.get("map"),
            "mAP75": seg.get("map75"),
            "precision_ultralytics": seg.get("mp"),
            "recall_ultralytics": seg.get("mr"),
        },
        "medical_macro": macro,
        "medical_global": analysis.get("medical_global", {}),
        "boundary_metric_units": analysis.get(
            "boundary_metric_units", {"asd": "mm", "hd95": "mm"}
        ),
        "boundary_status_counts": analysis.get("boundary_status_counts", {}),
        "boundary_valid_counts": analysis.get("boundary_valid_counts", {}),
        "boundary_distributions": {
            "asd": asd_dist,
            "hd95": hd95_dist,
        },
        "efficiency": profile,
        "speed": {
            "protocol": {
                "imgsz": args.imgsz,
                "batch": 1,
                "precision": args.precision,
                "warmup": args.warmup,
                "iterations": args.iters,
                "cuda_sync": True,
                "disk_io_included": False,
                "representative_image": str(benchmark_image),
            },
            "network_forward": net,
            "end_to_end": e2e,
            "real_time_definition": "E2E FPS >= 30 and mean latency <= 33.33 ms",
            "real_time_pass": real_time_pass,
        },
        "hardware": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device": args.device,
            "gpu_name": (
                torch.cuda.get_device_name(
                    int(args.device) if args.device.isdigit() else 0
                )
                if torch.cuda.is_available() and args.device.lower() != "cpu"
                else None
            ),
            "ultralytics_path": str(loaded_ultra),
        },
        "raw_official_val": official,
        "analysis_summary_path": str(analysis_path),
        "speed_json_path": str(speed_json),
    }

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_out = out.with_suffix(".md")
    md = f"""# {args.method} evaluation metrics

| Metric | Value |
|---|---:|
| Mask mAP50 | {_fmt(payload['official_mask_metrics']['mAP50'], 6)} |
| Mask mAP50-95 | {_fmt(payload['official_mask_metrics']['mAP50_95'], 6)} |
| Mask mAP75 | {_fmt(payload['official_mask_metrics']['mAP75'], 6)} |
| Dice | {_fmt(macro.get('dice'), 6)} |
| IoU | {_fmt(macro.get('iou'), 6)} |
| Precision | {_fmt(macro.get('precision'), 6)} |
| Sensitivity | {_fmt(macro.get('sensitivity'), 6)} |
| Specificity | {_fmt(macro.get('specificity'), 6)} |
| ASD (mm), mean ± std | {_fmt(macro.get('asd'), 6)} ± {_fmt(asd_dist.get('std'), 6)} |
| ASD valid / excluded | {asd_dist.get('valid_n', 'N/A')} / {asd_dist.get('excluded_n', 'N/A')} |
| HD95 (mm), mean ± std | {_fmt(macro.get('hd95'), 6)} ± {_fmt(hd95_dist.get('std'), 6)} |
| HD95 valid / excluded | {hd95_dist.get('valid_n', 'N/A')} / {hd95_dist.get('excluded_n', 'N/A')} |
| Full graph Params (M) | {_fmt(profile.get('full_training_graph_params_M'), 3)} |
| Fused deployment Params (M) | {_fmt(profile.get('deployment_fused_params_M'), 3)} |
| Deployment GMACs | {_fmt(profile.get('deployment_GMACs'), 3)} |
| Deployment GFLOPs | {_fmt(profile.get('deployment_GFLOPs'), 3)} |
| Checkpoint size (MB) | {_fmt(profile.get('checkpoint_size_MB'), 2)} |
| Network mean latency (ms) | {_fmt(net.get('mean_ms'), 3)} |
| Network FPS | {_fmt(net.get('fps_from_mean'), 2)} |
| E2E mean latency (ms) | {_fmt(e2e.get('mean_ms'), 3)} |
| E2E p95 latency (ms) | {_fmt(e2e.get('p95_ms'), 3)} |
| E2E FPS | {_fmt(e2e.get('fps_from_mean'), 2)} |
| Real-time >= 30 FPS | {'PASS' if real_time_pass else 'FAIL'} |

## Protocol

- Input size: {args.imgsz}×{args.imgsz}
- Validation batch: {args.batch}
- Medical mask confidence threshold: {args.conf}
- Latency batch: 1
- Precision: {args.precision}
- Warm-up: {args.warmup}
- Timed iterations: {args.iters}
- CUDA synchronization: yes
- Disk I/O included in E2E timing: no
- Boundary metrics: ASD and HD95 in mm using the same HEAL-style unit-spacing protocol as the proposed-method evaluator.
- Empty boundary cases: ASD/HD95 are NaN and excluded from boundary aggregation; counts are reported separately. Overlap metrics still include every image.
- GPU: {payload['hardware']['gpu_name']}
"""
    md_out.write_text(md, encoding="utf-8")

    print("\n" + "=" * 88)
    print(f"FULL SEGMENTATION EVALUATION — {args.method}")
    print("=" * 88)
    print(f"Mask mAP50    : {_fmt(payload['official_mask_metrics']['mAP50'], 6)}")
    print(f"Mask mAP50-95 : {_fmt(payload['official_mask_metrics']['mAP50_95'], 6)}")
    print(f"Mask mAP75    : {_fmt(payload['official_mask_metrics']['mAP75'], 6)}")
    print(f"Dice          : {_fmt(macro.get('dice'), 6)}")
    print(f"IoU           : {_fmt(macro.get('iou'), 6)}")
    print(f"Precision     : {_fmt(macro.get('precision'), 6)}")
    print(f"Sensitivity   : {_fmt(macro.get('sensitivity'), 6)}")
    print(f"Specificity   : {_fmt(macro.get('specificity'), 6)}")
    print(
        f"ASD           : {_fmt(macro.get('asd'), 3)} ± "
        f"{_fmt(asd_dist.get('std'), 3)} mm "
        f"(valid={asd_dist.get('valid_n', 'N/A')}, "
        f"excluded={asd_dist.get('excluded_n', 'N/A')})"
    )
    print(
        f"HD95          : {_fmt(macro.get('hd95'), 3)} ± "
        f"{_fmt(hd95_dist.get('std'), 3)} mm "
        f"(valid={hd95_dist.get('valid_n', 'N/A')}, "
        f"excluded={hd95_dist.get('excluded_n', 'N/A')})"
    )
    print(f"Params full   : {_fmt(profile.get('full_training_graph_params_M'), 3)} M")
    print(f"Params fused  : {_fmt(profile.get('deployment_fused_params_M'), 3)} M")
    print(f"GMACs         : {_fmt(profile.get('deployment_GMACs'), 3)}")
    print(f"GFLOPs        : {_fmt(profile.get('deployment_GFLOPs'), 3)}")
    print(f"Checkpoint    : {_fmt(profile.get('checkpoint_size_MB'), 2)} MB")
    print(f"E2E latency   : {_fmt(e2e.get('mean_ms'), 3)} ms")
    print(f"E2E FPS       : {_fmt(e2e.get('fps_from_mean'), 2)}")
    print(f"Real-time     : {'PASS' if real_time_pass else 'FAIL'}")
    print("Saved JSON    :", out)
    print("Saved table   :", md_out)
    print("Artifacts     :", artifact_dir)


if __name__ == "__main__":
    main()
