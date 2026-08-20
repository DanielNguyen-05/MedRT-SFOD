#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

# IMPORTANT:
# Put the MedRT-SFOD repository ahead of site-packages BEFORE importing
# Ultralytics. The project uses a local YOLO26 implementation whose BaseModel
# supports the dual-head segmentation outputs required by BDL.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import ultralytics
from ultralytics import YOLO

from boundary_dhf_seg import generate_boundary_dhf_pseudo_masks  # noqa: E402

EPS = 1e-8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def div(a: float, b: float) -> float:
    return float(a / (b + EPS))


def resolve_device(x: str) -> torch.device:
    if x.lower() == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")
    if x.isdigit():
        return torch.device(f"cuda:{x}")
    return torch.device(x)


def list_images(folder: Path) -> list[Path]:
    return sorted(
        x for x in folder.iterdir()
        if x.is_file() and x.suffix.lower() in IMAGE_EXTS
    )


def filter_images(paths: list[Path], ids: list[str] | None) -> list[Path]:
    if not ids:
        return paths
    wanted = {Path(x).stem for x in ids}
    out = [p for p in paths if p.stem in wanted]
    missing = wanted - {p.stem for p in out}
    if missing:
        print("[WARN] not found:", ", ".join(sorted(missing)))
    return out


def clean_letterbox(
    bgr: np.ndarray,
    imgsz: int,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = imgsz / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top = (imgsz - nh) // 2
    left = (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    tensor = (
        torch.from_numpy(np.ascontiguousarray(canvas))
        .permute(2, 0, 1).float() / 255.0
    )
    return tensor, {
        "orig_h": h, "orig_w": w,
        "nh": nh, "nw": nw,
        "top": top, "left": left,
        "imgsz": imgsz,
    }


def restore_mask(
    x: torch.Tensor,
    meta: dict[str, int | float],
    binary: bool,
) -> np.ndarray:
    x = x.detach().float().cpu()
    if x.ndim != 2:
        raise ValueError(f"Expected HxW, got {tuple(x.shape)}")

    imgsz = int(meta["imgsz"])
    y = x[None, None]
    if tuple(y.shape[-2:]) != (imgsz, imgsz):
        if binary:
            y = F.interpolate(y, (imgsz, imgsz), mode="nearest")
        else:
            y = F.interpolate(
                y, (imgsz, imgsz),
                mode="bilinear", align_corners=False,
            )

    top, left = int(meta["top"]), int(meta["left"])
    nh, nw = int(meta["nh"]), int(meta["nw"])
    crop = y[0, 0, top:top + nh, left:left + nw].numpy()

    interp = cv2.INTER_NEAREST if binary else cv2.INTER_LINEAR
    out = cv2.resize(
        crop,
        (int(meta["orig_w"]), int(meta["orig_h"])),
        interpolation=interp,
    )
    if binary:
        return (out > 0.5).astype(np.uint8)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def union_teacher_masks(masks: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    if masks is None or masks.numel() == 0 or masks.shape[0] == 0:
        return torch.zeros(shape, dtype=torch.float32)
    return masks.float().amax(dim=0)


def student_union(result, h: int, w: int) -> np.ndarray:
    if (
        result.masks is None
        or result.masks.data is None
        or result.masks.data.numel() == 0
    ):
        return np.zeros((h, w), dtype=np.uint8)

    m = result.masks.data.detach().float()
    if m.ndim == 2:
        m = m.unsqueeze(0)
    y = m.amax(dim=0, keepdim=True).unsqueeze(0)
    if tuple(y.shape[-2:]) != (h, w):
        y = F.interpolate(y, (h, w), mode="nearest")
    return (y[0, 0] > 0.5).cpu().numpy().astype(np.uint8)


def overlap(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | int]:
    p, g = pred.astype(bool), gt.astype(bool)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    return {
        "dice": div(2 * tp, 2 * tp + fp + fn),
        "iou": div(tp, tp + fp + fn),
        "precision": div(tp, tp + fp),
        "sensitivity": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def surface(mask: np.ndarray) -> np.ndarray:
    x = (mask > 0).astype(np.uint8)
    if not x.any():
        return np.zeros_like(x, dtype=bool)
    er = cv2.erode(x, np.ones((3, 3), np.uint8), iterations=1)
    return np.logical_and(x.astype(bool), ~er.astype(bool))


def surface_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float | str]:
    p, g = pred.astype(bool), gt.astype(bool)
    diag = float(math.hypot(*p.shape))
    if not p.any() and not g.any():
        return {"asd": 0.0, "hd95": 0.0, "status": "both_empty"}
    if not p.any() or not g.any():
        return {
            "asd": diag, "hd95": diag,
            "status": "pred_empty" if not p.any() else "gt_empty",
        }

    ps, gs = surface(p), surface(g)
    if not ps.any() or not gs.any():
        return {"asd": diag, "hd95": diag, "status": "surface_empty"}

    dt_g = cv2.distanceTransform(
        (~gs).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    dt_p = cv2.distanceTransform(
        (~ps).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    d = np.concatenate([dt_g[ps], dt_p[gs]]).astype(np.float64)
    return {
        "asd": float(d.mean()),
        "hd95": float(np.percentile(d, 95)),
        "status": "ok",
    }


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    p, g = pred.astype(bool), gt.astype(bool)
    out = np.zeros(gt.shape, np.uint8)
    out[np.logical_and(p, g)] = 1
    out[np.logical_and(p, ~g)] = 2
    out[np.logical_and(~p, g)] = 3
    return out


def save_mask(path: Path, x: np.ndarray, binary: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        y = (x > 0.5).astype(np.uint8) * 255
    else:
        y = np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(path), y)


def save_panel(
    out: Path,
    bgr: np.ndarray,
    gt: np.ndarray,
    o2o: np.ndarray,
    o2m: np.ndarray,
    fused: np.ndarray,
    soft: np.ndarray,
    dis: np.ndarray,
    bnd: np.ndarray,
    student: np.ndarray,
    err: np.ndarray,
    row: dict[str, Any],
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    items = [
        (rgb, "Image", None),
        (gt, "GT", (0, 1)),
        (o2o, "Initial Teacher O2O", (0, 1)),
        (o2m, "Teacher O2M witnesses", (0, 1)),
        (fused, "Teacher fused hard pseudo", (0, 1)),
        (soft, "Teacher soft target", (0, 1)),
        (dis, "BDL disagreement", (0, 1)),
        (bnd, "BDL inner boundary", (0, 1)),
        (student, "Final Student prediction", (0, 1)),
        (err, "Student error: 1 TP / 2 FP / 3 FN", (0, 3)),
    ]
    for ax, (arr, title, lim) in zip(axes.flat, items):
        if lim is None:
            ax.imshow(arr)
        else:
            ax.imshow(arr, vmin=lim[0], vmax=lim[1])
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{row['image']} | "
        f"O2O Dice={row['initial_o2o_dice']:.4f} | "
        f"Fused Dice={row['initial_fused_dice']:.4f} | "
        f"Student Dice={row['final_student_dice']:.4f} | "
        f"Fusion Δ={row['fusion_delta_dice']:+.4f} | "
        f"Self-train Δ={row['self_training_delta_dice']:+.4f}\n"
        f"Teacher fused ASD/HD95="
        f"{row['initial_fused_asd']:.2f}/{row['initial_fused_hd95']:.2f}px | "
        f"Student ASD/HD95="
        f"{row['final_student_asd']:.2f}/{row['final_student_hd95']:.2f}px"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt-masks", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--student-conf", type=float, default=0.25)
    ap.add_argument("--image-ids", nargs="*", default=None)
    ap.add_argument("--max-images", type=int, default=0)

    ap.add_argument("--tau-o2o", type=float, default=0.5)
    ap.add_argument("--tau-o2m", type=float, default=0.5)
    ap.add_argument("--tau-no", type=float, default=0.2)
    ap.add_argument("--tau-dup", type=float, default=0.7)
    ap.add_argument("--bdl-tau-match", type=float, default=0.5)
    ap.add_argument("--bdl-max-witnesses", type=int, default=5)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--stability-low", type=float, default=0.40)
    ap.add_argument("--stability-high", type=float, default=0.60)
    ap.add_argument("--mask-rel-thr", type=float, default=0.744898)
    ap.add_argument("--min-mask-pixels", type=int, default=16)
    ap.add_argument("--bdl-boundary-kernel", type=int, default=3)
    args = ap.parse_args()

    teacher_path = Path(args.teacher).resolve()
    student_path = Path(args.student).resolve()
    image_dir = Path(args.images).resolve()
    gt_dir = Path(args.gt_masks).resolve()
    out_dir = Path(args.out_dir).resolve()

    if not teacher_path.exists():
        raise FileNotFoundError(teacher_path)
    if not student_path.exists():
        raise FileNotFoundError(student_path)

    paths = filter_images(list_images(image_dir), args.image_ids)
    if args.max_images > 0:
        paths = paths[:args.max_images]
    if not paths:
        raise RuntimeError("No images selected")

    device = resolve_device(args.device)

    print("=" * 88)
    print("MEDRT-SFSEG INITIAL TEACHER vs FINAL STUDENT TRACE")
    print("=" * 88)
    print("Teacher      :", teacher_path)
    print("Student      :", student_path)
    print("Ultralytics  :", Path(ultralytics.__file__).resolve())
    local_ultra = (REPO_ROOT / "ultralytics").resolve()
    loaded_ultra = Path(ultralytics.__file__).resolve()
    if local_ultra not in loaded_ultra.parents:
        raise RuntimeError(
            "Wrong Ultralytics package loaded. Expected local repo package under "
            f"{local_ultra}, but got {loaded_ultra}. "
            "Run from ~/MedRT-SFOD with PYTHONPATH=$PWD."
        )
    print("Images       :", len(paths))
    print("Teacher view : deterministic clean letterbox")
    print("GT usage     : post-training analysis only")
    print("Retraining   : NO")
    print("Output       :", out_dir)
    print()

    tw = YOLO(str(teacher_path), task="segment")
    teacher = tw.model.to(device).float().eval()
    for p in teacher.parameters():
        p.requires_grad = False
    head = teacher.model[-1]
    if not hasattr(head, "one2one") or not hasattr(head, "one2many"):
        raise RuntimeError("Teacher checkpoint is not an unfused dual-head model")

    student_model = YOLO(str(student_path), task="segment")

    dirs = {
        "gt": out_dir / "masks/gt",
        "o2o": out_dir / "masks/teacher_o2o",
        "o2m": out_dir / "masks/teacher_o2m_witness",
        "fused": out_dir / "masks/teacher_fused_hard",
        "soft": out_dir / "masks/teacher_soft_target",
        "dis": out_dir / "masks/bdl_disagreement",
        "bnd": out_dir / "masks/bdl_boundary",
        "student": out_dir / "masks/student_final",
        "err": out_dir / "masks/student_error",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    panel_dir = out_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for i, image_path in enumerate(paths, 1):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_path)

        gt_path = gt_dir / f"{image_path.stem}.png"
        gt0 = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if gt0 is None:
            raise FileNotFoundError(gt_path)
        gt = (gt0 > 0).astype(np.uint8)

        x_cpu, meta = clean_letterbox(bgr, args.imgsz)
        weak = x_cpu.unsqueeze(0).to(device)

        (
            labels_out,
            supervision_masks_out,
            geometry_masks_out,
            reliability_out,
            totals,
            debug_out,
        ) = generate_boundary_dhf_pseudo_masks(
            teacher=teacher,
            weak_imgs=weak,
            tau_o2o=args.tau_o2o,
            tau_o2m=args.tau_o2m,
            tau_no=args.tau_no,
            tau_dup=args.tau_dup,
            tau_match=args.bdl_tau_match,
            max_witnesses=args.bdl_max_witnesses,
            mask_threshold=args.mask_thr,
            stability_low=args.stability_low,
            stability_high=args.stability_high,
            reliability_threshold=args.mask_rel_thr,
            min_mask_pixels=args.min_mask_pixels,
            boundary_kernel=args.bdl_boundary_kernel,
            return_debug=True,
        )

        dbg = debug_out[0]
        o2o = restore_mask(dbg["o2o_hard"], meta, True)
        o2m = restore_mask(dbg["o2m_witness"], meta, False)
        soft = restore_mask(dbg["soft_target"], meta, False)
        dis = restore_mask(dbg["disagreement"], meta, False)
        bnd = restore_mask(dbg["boundary"], meta, True)

        gm = geometry_masks_out[0]
        if gm.numel() == 0 or gm.shape[0] == 0:
            proto_shape = tuple(dbg["o2o_hard"].shape[-2:])
            fused_proto = torch.zeros(proto_shape, dtype=torch.float32)
        else:
            fused_proto = union_teacher_masks(gm, tuple(gm.shape[-2:]))
        fused = restore_mask(fused_proto, meta, True)

        sr = student_model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.student_conf,
            device=args.device,
            verbose=False,
        )[0]
        student = student_union(sr, gt.shape[0], gt.shape[1])

        mo = overlap(o2o, gt)
        mf = overlap(fused, gt)
        ms = overlap(student, gt)
        mts = overlap(student, fused)

        bo = surface_metrics(o2o, gt)
        bf = surface_metrics(fused, gt)
        bs = surface_metrics(student, gt)

        err = error_map(student, gt)

        row = {
            "image": image_path.name,

            "initial_o2o_dice": mo["dice"],
            "initial_o2o_iou": mo["iou"],
            "initial_o2o_asd": bo["asd"],
            "initial_o2o_hd95": bo["hd95"],

            "initial_fused_dice": mf["dice"],
            "initial_fused_iou": mf["iou"],
            "initial_fused_precision": mf["precision"],
            "initial_fused_sensitivity": mf["sensitivity"],
            "initial_fused_specificity": mf["specificity"],
            "initial_fused_asd": bf["asd"],
            "initial_fused_hd95": bf["hd95"],
            "initial_fused_boundary_status": bf["status"],

            "final_student_dice": ms["dice"],
            "final_student_iou": ms["iou"],
            "final_student_precision": ms["precision"],
            "final_student_sensitivity": ms["sensitivity"],
            "final_student_specificity": ms["specificity"],
            "final_student_asd": bs["asd"],
            "final_student_hd95": bs["hd95"],
            "final_student_boundary_status": bs["status"],

            "teacher_student_dice": mts["dice"],
            "teacher_student_iou": mts["iou"],

            "fusion_delta_dice": mf["dice"] - mo["dice"],
            "fusion_delta_iou": mf["iou"] - mo["iou"],
            "self_training_delta_dice": ms["dice"] - mf["dice"],
            "self_training_delta_iou": ms["iou"] - mf["iou"],
            "self_training_delta_asd": bs["asd"] - bf["asd"],
            "self_training_delta_hd95": bs["hd95"] - bf["hd95"],

            "teacher_anchors": int(totals.get("anchors", 0)),
            "teacher_o2m_candidates": int(totals.get("candidates", 0)),
            "teacher_bdl_anchors": int(totals.get("bdl_anchors", 0)),
            "teacher_bdl_witnesses": int(totals.get("bdl_witnesses", 0)),
            "teacher_box_extras": int(totals.get("box_dhf_extras", 0)),
            "teacher_mask_extras": int(totals.get("mask_dhf_extras", 0)),
            "teacher_rejected_reliability": int(
                totals.get("rejected_reliability", 0)
            ),
            "teacher_pseudo_instances": int(totals.get("pseudo", 0)),
            "teacher_boundary_disagreement": float(
                totals.get("bdl_boundary_disagreement_mean", 0.0)
            ),
            "teacher_interior_disagreement": float(
                totals.get("bdl_interior_disagreement_mean", 0.0)
            ),
            "teacher_boundary_interior_ratio": float(
                totals.get("bdl_boundary_interior_ratio", 0.0)
            ),
            "teacher_soft_shift": float(
                totals.get("bdl_soft_target_shift_mean", 0.0)
            ),
        }
        rows.append(row)

        save_mask(dirs["gt"] / f"{image_path.stem}.png", gt, True)
        save_mask(dirs["o2o"] / f"{image_path.stem}.png", o2o, True)
        save_mask(dirs["o2m"] / f"{image_path.stem}.png", o2m, False)
        save_mask(dirs["fused"] / f"{image_path.stem}.png", fused, True)
        save_mask(dirs["soft"] / f"{image_path.stem}.png", soft, False)
        save_mask(dirs["dis"] / f"{image_path.stem}.png", dis, False)
        save_mask(dirs["bnd"] / f"{image_path.stem}.png", bnd, True)
        save_mask(dirs["student"] / f"{image_path.stem}.png", student, True)
        save_mask(dirs["err"] / f"{image_path.stem}.png", err / 3.0, False)

        save_panel(
            panel_dir / f"{image_path.stem}.png",
            bgr, gt, o2o, o2m, fused, soft, dis, bnd, student, err, row,
        )

        print(
            f"[{i:04d}/{len(paths):04d}] {image_path.name} | "
            f"O2O={row['initial_o2o_dice']:.4f} "
            f"Fused={row['initial_fused_dice']:.4f} "
            f"Student={row['final_student_dice']:.4f} "
            f"FusionΔ={row['fusion_delta_dice']:+.4f} "
            f"SelfTrainΔ={row['self_training_delta_dice']:+.4f}"
        )

    write_csv(out_dir / "per_image_trace.csv", rows)

    summary = {
        "teacher_checkpoint": str(teacher_path),
        "student_checkpoint": str(student_path),
        "num_images": len(rows),
        "teacher_role": "initial AdaBN Teacher",
        "student_role": "final adapted Student",
        "teacher_view": "deterministic clean centered letterbox",
        "historical_training_pseudo_exact_replay": False,
        "gt_usage": "post-training analysis only",
        "retraining_performed": False,
        "macro": {
            "initial_o2o_dice": mean(rows, "initial_o2o_dice"),
            "initial_o2o_iou": mean(rows, "initial_o2o_iou"),
            "initial_fused_dice": mean(rows, "initial_fused_dice"),
            "initial_fused_iou": mean(rows, "initial_fused_iou"),
            "initial_fused_asd": mean(rows, "initial_fused_asd"),
            "initial_fused_hd95": mean(rows, "initial_fused_hd95"),
            "final_student_dice": mean(rows, "final_student_dice"),
            "final_student_iou": mean(rows, "final_student_iou"),
            "final_student_asd": mean(rows, "final_student_asd"),
            "final_student_hd95": mean(rows, "final_student_hd95"),
            "teacher_student_dice": mean(rows, "teacher_student_dice"),
            "fusion_delta_dice": mean(rows, "fusion_delta_dice"),
            "self_training_delta_dice": mean(rows, "self_training_delta_dice"),
            "self_training_delta_asd": mean(rows, "self_training_delta_asd"),
            "self_training_delta_hd95": mean(rows, "self_training_delta_hd95"),
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    m = summary["macro"]
    print()
    print("=" * 88)
    print("TRACE SUMMARY")
    print("=" * 88)
    print(f"Initial Teacher O2O Dice  : {m['initial_o2o_dice']:.6f}")
    print(f"Initial Teacher Fused Dice: {m['initial_fused_dice']:.6f}")
    print(f"Final Student Dice        : {m['final_student_dice']:.6f}")
    print(f"Mean fusion Δ Dice        : {m['fusion_delta_dice']:+.6f}")
    print(f"Mean self-training Δ Dice : {m['self_training_delta_dice']:+.6f}")
    print(f"Teacher Fused ASD         : {m['initial_fused_asd']:.6f} px")
    print(f"Final Student ASD         : {m['final_student_asd']:.6f} px")
    print(f"Teacher Fused HD95        : {m['initial_fused_hd95']:.6f} px")
    print(f"Final Student HD95        : {m['final_student_hd95']:.6f} px")
    print("Saved panels              :", panel_dir)
    print("Saved CSV                 :", out_dir / "per_image_trace.csv")
    print("Saved summary             :", out_dir / "summary.json")
    print("[PASS] Completed without retraining")


if __name__ == "__main__":
    main()
