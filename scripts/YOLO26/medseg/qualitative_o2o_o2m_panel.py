#!/usr/bin/env python3
"""
Qualitative O2O -> O2M -> disagreement -> final-prediction panel figure.

Produces the "3-4 representative example" figure requested alongside the
hallucination-suppression comparison: for each selected image, shows the
Teacher's O2O anchor mask, the aggregated O2M witness evidence, the signed
O2O/O2M disagreement map (the actual DURR routing signal), and the final
adapted Student prediction (optionally with the GT boundary overlaid for
qualitative sanity-checking only -- GT is never used to pick images or to
drive the pipeline).

Reuses the exact same DURR pseudo-mask construction
(`durr_seg.generate_durr_pseudo_masks`) and letterbox/restore helpers already
used by the automatic post-training trace in `stage2_dense_sfseg_durr.py`, so
the panels are pixel-for-pixel consistent with what training actually saw.

EVALUATION-ONLY. Does not modify or re-run adaptation.

Example
-------
python qualitative_o2o_o2m_panel.py \
  --teacher-weights runs/seg/dense_sfseg/cvc_durr_v1_60ep/checkpoints/durr_teacher_ema_epoch_60.pt \
  --student-weights runs/seg/dense_sfseg/cvc_durr_v1_60ep/checkpoints/durr_student_epoch_60.pt \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --out-dir runs/seg/analysis/qualitative/k2c_durr \
  --num-examples 4 --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from durr_seg import generate_durr_pseudo_masks  # noqa: E402
from stage2_dense_sfseg_durr import _analysis_letterbox, _analysis_restore  # noqa: E402
from smoke_stage2_mt_seg import resolve_device, list_images  # noqa: E402


def _union_mask(result, h: int, w: int) -> np.ndarray:
    if result.masks is None or result.masks.data is None or result.masks.data.numel() == 0:
        return np.zeros((h, w), dtype=np.uint8)
    masks = result.masks.data.detach().float()
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    merged = masks.amax(dim=0, keepdim=True).unsqueeze(0)
    if tuple(merged.shape[-2:]) != (h, w):
        merged = F.interpolate(merged, size=(h, w), mode="nearest")
    return (merged[0, 0] > 0.5).cpu().numpy().astype(np.uint8)


def _boundary_overlay(image_rgb: np.ndarray, gt: np.ndarray | None, pred: np.ndarray) -> np.ndarray:
    vis = image_rgb.copy()
    pred_u8 = (pred > 0).astype(np.uint8)
    contours_p, _ = cv2.findContours(pred_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours_p, -1, (255, 0, 0), 2)  # red = prediction
    if gt is not None:
        gt_u8 = (gt > 0).astype(np.uint8)
        contours_g, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_g, -1, (0, 255, 0), 2)  # green = GT
    return vis


def build_routes(teacher, x: torch.Tensor, args) -> dict:
    (_labels, _masks, _rel, _ps, routes) = generate_durr_pseudo_masks(
        teacher=teacher,
        weak_imgs=x,
        tau_o2o=args.tau_o2o,
        tau_o2m=args.tau_o2m,
        tau_no=args.tau_no,
        tau_dup=args.tau_dup,
        tau_match=args.durr_tau_match,
        max_witnesses=args.durr_max_witnesses,
        mask_threshold=args.mask_thr,
        stability_low=args.stability_low,
        stability_high=args.stability_high,
        reliability_threshold=args.mask_rel_thr,
        min_mask_pixels=args.min_mask_pixels,
        boundary_kernel=args.durr_boundary_kernel,
        route_gain=args.durr_route_gain,
        route_min_disagreement=args.durr_min_disagreement,
        rescue_conf=args.durr_rescue_conf,
        rescue_stability=args.durr_rescue_stability,
        rescue_consensus_iou=args.durr_rescue_consensus_iou,
        rescue_min_support=args.durr_rescue_min_support,
        evidence_conf=args.durr_evidence_conf,
        safe_bg_teacher_prob=args.durr_safe_bg_teacher_prob,
    )
    return routes[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-weights", required=True)
    ap.add_argument("--student-weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--gt-masks", default=None, help="Optional; overlay only, never used for selection.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25, help="Student prediction confidence.")
    ap.add_argument("--num-examples", type=int, default=4)
    ap.add_argument(
        "--image-list", default=None,
        help="Optional comma-separated image filenames to use instead of auto-selection.",
    )
    ap.add_argument(
        "--rank-pool-size", type=int, default=0,
        help="0 = scan the full target set to rank by |signed delta| (slower, most faithful). "
             "N>0 = scan only the first N images (sorted) for speed.",
    )

    # Mask-DHF base (defaults match stage2_dense_sfseg_durr.py).
    ap.add_argument("--tau-o2o", type=float, default=0.5)
    ap.add_argument("--tau-o2m", type=float, default=0.5)
    ap.add_argument("--tau-no", type=float, default=0.2)
    ap.add_argument("--tau-dup", type=float, default=0.7)
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--stability-low", type=float, default=0.40)
    ap.add_argument("--stability-high", type=float, default=0.60)
    ap.add_argument("--mask-rel-thr", type=float, default=0.744898)
    ap.add_argument("--min-mask-pixels", type=int, default=16)

    # DURR matching / routing (defaults match stage2_dense_sfseg_durr.py).
    ap.add_argument("--durr-tau-match", type=float, default=0.5)
    ap.add_argument("--durr-max-witnesses", type=int, default=5)
    ap.add_argument("--durr-boundary-kernel", type=int, default=5)
    ap.add_argument("--durr-route-gain", type=float, default=1.0)
    ap.add_argument("--durr-min-disagreement", type=float, default=0.02)
    ap.add_argument("--durr-rescue-conf", type=float, default=0.80)
    ap.add_argument("--durr-rescue-stability", type=float, default=0.80)
    ap.add_argument("--durr-rescue-consensus-iou", type=float, default=0.70)
    ap.add_argument("--durr-rescue-min-support", type=int, default=0)
    ap.add_argument("--durr-evidence-conf", type=float, default=0.10)
    ap.add_argument("--durr-safe-bg-teacher-prob", type=float, default=0.10)

    args = ap.parse_args()

    device = resolve_device(args.device)
    out_dir = Path(args.out_dir).resolve()
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)

    teacher_wrapper = YOLO(args.teacher_weights)
    teacher = teacher_wrapper.model.to(device).float().eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student_wrapper = YOLO(args.student_weights)

    images = list_images(Path(args.images).resolve())
    gt_dir = Path(args.gt_masks).resolve() if args.gt_masks else None

    if args.image_list:
        wanted = {n.strip() for n in args.image_list.split(",")}
        selected = [p for p in images if p.name in wanted]
        if not selected:
            raise RuntimeError(f"None of --image-list names matched files in {args.images}")
    else:
        pool = images if args.rank_pool_size <= 0 else images[: args.rank_pool_size]
        print(f"[RANK] scoring {len(pool)} images by mean(|signed delta|) to auto-select "
              f"the {args.num_examples} strongest-disagreement examples...")
        scored: list[tuple[float, Path]] = []
        for i, image_path in enumerate(pool, 1):
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            x_cpu, _meta = _analysis_letterbox(bgr, args.imgsz)
            x = x_cpu.unsqueeze(0).to(device)
            with torch.no_grad():
                route = build_routes(teacher, x, args)
            score = float(route["signed_delta"].abs().mean().item())
            scored.append((score, image_path))
            if i % 100 == 0 or i == len(pool):
                print(f"  [{i}/{len(pool)}]")
        scored.sort(key=lambda t: t[0], reverse=True)
        selected = [p for _, p in scored[: args.num_examples]]
        (out_dir / "auto_selected_ranking.csv").write_text(
            "image,mean_abs_signed_delta\n"
            + "\n".join(f"{p.name},{s:.6f}" for s, p in scored[: max(50, args.num_examples)]),
            encoding="utf-8",
        )

    print("[SELECTED]", [p.name for p in selected])

    panel_paths: list[Path] = []
    for image_path in selected:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x_cpu, meta = _analysis_letterbox(bgr, args.imgsz)
        x = x_cpu.unsqueeze(0).to(device)

        with torch.no_grad():
            route = build_routes(teacher, x, args)

        o2o = _analysis_restore(route["o2o_union"], meta, binary=True)
        o2m = _analysis_restore(route["o2m_witness"], meta, binary=False)
        delta = _analysis_restore(route["signed_delta"], meta, binary=False, clip_range=None)

        result = student_wrapper.predict(
            source=str(image_path), imgsz=args.imgsz, conf=args.conf,
            device=args.device, verbose=False,
        )[0]
        h, w = rgb.shape[:2]
        student_mask = _union_mask(result, h, w)

        gt = None
        if gt_dir is not None:
            gt_raw = cv2.imread(str(gt_dir / f"{image_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
            if gt_raw is not None:
                gt = (gt_raw > 0).astype(np.uint8)

        overlay = _boundary_overlay(rgb, gt, student_mask)

        fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
        items = [
            (rgb, "Image", None, None),
            (o2o, "Teacher O2O anchor", "gray", (0, 1)),
            (o2m, "Teacher O2M witness (aggregated)", "gray", (0, 1)),
            (delta, "Signed disagreement Δ = P_m − P_o\n(warm=expand, cool=shrink)", "coolwarm", (-1, 1)),
            (overlay, "Final Student (red) vs GT (green)" if gt is not None else "Final Student prediction", None, None),
        ]
        for ax, (arr, title, cmap, lim) in zip(axes, items):
            if lim is None:
                ax.imshow(arr)
            else:
                ax.imshow(arr, cmap=cmap, vmin=lim[0], vmax=lim[1])
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        fig.suptitle(image_path.name)
        fig.tight_layout()
        out_path = out_dir / "panels" / f"{image_path.stem}.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        panel_paths.append(out_path)
        print("[SAVED]", out_path)

    # Combined contact sheet: stack all selected panels vertically.
    if panel_paths:
        imgs = [cv2.imread(str(p)) for p in panel_paths]
        widths = [im.shape[1] for im in imgs]
        target_w = max(widths)
        resized = []
        for im in imgs:
            if im.shape[1] != target_w:
                scale = target_w / im.shape[1]
                im = cv2.resize(im, (target_w, int(round(im.shape[0] * scale))))
            resized.append(im)
        contact_sheet = np.concatenate(resized, axis=0)
        contact_path = out_dir / "contact_sheet.png"
        cv2.imwrite(str(contact_path), contact_sheet)
        print("[SAVED]", contact_path)

    print("[PASS] Qualitative panel generation completed")


if __name__ == "__main__":
    main()
