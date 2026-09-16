# Ablation / Evaluation Experiments — Advisor Feedback Round

Companion to `README.md`. This file gives copy-paste commands for the four
experiment groups your advisor asked for. It assumes the same environment
setup as `README.md` §0 (`source .venv/bin/activate; export PYTHONPATH="$PWD"`)
and the same canonical checkpoint paths already established there.

New/modified files this round:

| File | What changed |
|---|---|
| `scripts/YOLO26/medseg/boundary_metrics.py` | **new** — Boundary IoU, BF-score, FP-region metrics |
| `scripts/YOLO26/medseg/evaluate_stratified.py` | **new** — multi-checkpoint, size-stratified, boundary-aware, FP/hallucination evaluator (drives items 1–3) |
| `scripts/YOLO26/medseg/qualitative_o2o_o2m_panel.py` | **new** — O2O → O2M → Δ → final panel figure (item 2's qualitative figure) |
| `scripts/YOLO26/medseg/durr_seg.py` | **patched** — `generate_durr_pseudo_masks(..., signed_mode=...)` sign-ablation switch |
| `scripts/YOLO26/medseg/stage2_dense_sfseg_durr.py` | **patched** — new `--durr-signed-mode` CLI flag, threaded through training + the post-training trace, recorded in `stage2_metadata.json` |

Item 5 (Params/GFLOPs/FPS/Dice for DPL/UPL-SFDA/HEAL and a Pareto plot) is
**not** included here — it needs three external repos reimplemented against
your exact eval protocol, which is a separate, much larger effort. Say the
word when you want to scope that one; I'd start by checking which of the
three have public code before writing anything.

---

## 0. Sign-ablation semantics (item 4) — please sanity-check this with your advisor

Your advisor's four bullets ("Magnitude only: `|Δ|`", "Unsigned disagreement:
`|Pm−Po|`", "Signed disagreement: `Pm−Po`", "Reliability-weighted signed
disagreement") are mathematically ambiguous as literally written (`|Δ|` and
`|Pm−Po|` are the same quantity). I operationalized them as four **distinct**
mechanisms so the ablation actually isolates something, via
`--durr-signed-mode {magnitude_only, unsigned, signed, signed_reliability}`:

| Mode | Weight | Directional target | Isolates |
|---|---|---|---|
| `magnitude_only` | `r_pair` (constant) | `P_o` (no shift at all) | The "uncertainty as filtering" baseline from Related Work §2.2 — disagreement only gates *where* to (re-)supervise, never *how*. |
| `unsigned` | `\|Δ\| · r_pair` | `clip(P_o + g·\|Δ\|)` — always expands | Uses the magnitude but throws away the sign (always pushes outward). |
| `signed` | `\|Δ\|` (no reliability factor) | `clip(P_o + g·Δ)` — sign preserved | Whether reliability weighting matters, independent of sign. |
| `signed_reliability` | `\|Δ\| · r_pair` | `clip(P_o + g·Δ)` — sign preserved | **Full/default formulation** — unchanged behavior, this is what all your existing runs used. |

All four modes share the exact same routing region (same boundary band,
same `route_min_disagreement` gate) — only the weight/target computation
differs. This is documented in `durr_seg.py`'s module docstring
(`DURR_SIGNED_MODES`). If your advisor meant something else by these four
labels, the fix is a one-line change in `durr_seg.py`'s
`generate_durr_pseudo_masks` — flag it and I'll adjust.

---

## 1. Sign-ablation training runs (item 4)

Same hyperparameters as your existing DURR command (README §19), varying
only `--durr-signed-mode`. Run all 4 × both directions (8 runs total, same
cost as your existing DURR runs).

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

for MODE in magnitude_only unsigned signed signed_reliability; do
  mkdir -p runs/seg/dense_sfseg logs/seg/ablation_sign

  nohup env PYTHONPATH="$PWD" \
  python scripts/YOLO26/medseg/stage2_dense_sfseg_durr.py \
    --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
    --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
    --out-dir runs/seg/dense_sfseg/cvc_sign_${MODE}_60ep \
    --imgsz 640 --batch 4 --workers 4 --epochs 60 --lr 1e-4 \
    --device 0 --seed 29 \
    --tau-o2o 0.5 --tau-o2m 0.5 --tau-no 0.2 --tau-dup 0.7 \
    --mask-thr 0.5 --stability-low 0.40 --stability-high 0.60 \
    --mask-rel-thr 0.744898 --min-mask-pixels 16 \
    --durr-tau-match 0.5 --durr-max-witnesses 5 --durr-boundary-kernel 5 \
    --durr-route-gain 1.0 --durr-min-disagreement 0.02 \
    --durr-rescue-conf 0.80 --durr-rescue-stability 0.80 \
    --durr-rescue-consensus-iou 0.70 --durr-rescue-min-support 0 \
    --durr-evidence-conf 0.10 --durr-safe-bg-teacher-prob 0.10 \
    --durr-hall-student-thr 0.80 --durr-hall-area-thr 0.10 --durr-hall-area-weight 0.25 \
    --durr-lambda-dir 0.10 --durr-lambda-rescue 0.20 --durr-lambda-hall 0.10 \
    --durr-warmup-epochs 5 \
    --durr-signed-mode ${MODE} \
    --mard-lambda0 0.05 --mard-lambda-max 0.20 --mard-gamma 1.0 --mard-alpha 1.0 --mard-beta 0.1 \
    --mard-warmup-epochs 5 --mard-gate-threshold 0.5 --mard-topk-boxes 15 \
    --mard-fg-points 8 --mard-bg-points 128 --mard-eta 12 --mard-box-conf 0.5 \
    --segmard-erode-kernel 1 --segmard-dilate-kernel 1 --segmard-hard-bg-ratio 0.5 \
    --save-interval 60 \
  > logs/seg/ablation_sign/cvc_sign_${MODE}_60ep.log 2>&1

  echo "DONE MODE=${MODE} K2C  PID_was=$!"
done
```

Repeat the same loop for CVC→Kvasir with `--weights
runs/seg/stage1/frozen/yolo26s_seg_kvasir_adabn.pt`, `--target-images
dataset/Kvasir-SEG-YOLO26/images/target`, `--mask-rel-thr 0.753071`, and
`--out-dir runs/seg/dense_sfseg/kvasir_sign_${MODE}_60ep`. Run sequentially
(not `nohup ... &` in a loop) unless you have GPU headroom for concurrent
jobs, since each is a full 60-epoch run.

Checkpoints land at
`runs/seg/dense_sfseg/cvc_sign_${MODE}_60ep/checkpoints/durr_student_epoch_60.pt`.

### Score the sign-ablation (Dice / ASD / BF-score, item 4's table)

```bash
python scripts/YOLO26/medseg/evaluate_stratified.py \
  --model magnitude_only=runs/seg/dense_sfseg/cvc_sign_magnitude_only_60ep/checkpoints/durr_student_epoch_60.pt \
  --model unsigned=runs/seg/dense_sfseg/cvc_sign_unsigned_60ep/checkpoints/durr_student_epoch_60.pt \
  --model signed=runs/seg/dense_sfseg/cvc_sign_signed_60ep/checkpoints/durr_student_epoch_60.pt \
  --model signed_reliability=runs/seg/dense_sfseg/cvc_sign_signed_reliability_60ep/checkpoints/durr_student_epoch_60.pt \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --out-dir runs/seg/analysis/sign_ablation/k2c \
  --imgsz 640 --conf 0.25 --device 0
```

`table0_overall.md` in the output dir is the Dice/ASD/BF-score/Boundary-IoU
table for item 4 (K2C direction). Repeat with the four C2K checkpoints and
`--images/--gt-masks` swapped to Kvasir for the second direction.

If `signed_reliability` beats the other three clearly on both Dice and
BF-score, that single table is your strongest novelty evidence in the paper.

---

## 2. Checkpoints needed for items 1–3 (size-stratified / hallucination / boundary)

You need four checkpoints per direction: **Baseline**, **Mask-DHF**,
**Mask-DHF+DURR** (no SegMARD), **Full**. Baseline already exists
(`runs/seg/source/frozen/...`). Train the other three explicitly with
SegMARD forced off/on so the checkpoints are unambiguous — note
`--mard-lambda0` defaults to **0.05** (SegMARD partially on by default) in
both stage-2 scripts, so "Mask-DHF only" needs it explicitly zeroed:

```bash
cd ~/MedRT-SFOD
source .venv/bin/activate
export PYTHONPATH="$PWD"

# --- Mask-DHF only (SegMARD forced off) ------------------------------------
mkdir -p runs/seg/dense_sfseg/cvc_maskdhf_only_60ep logs/seg/ablation
python scripts/YOLO26/medseg/stage2_dense_sfseg.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/dense_sfseg/cvc_maskdhf_only_60ep \
  --imgsz 640 --batch 4 --workers 4 --epochs 60 --lr 1e-4 --grad-clip 10 --ema 0.999 \
  --dhf-mode mask \
  --tau-o2o 0.5 --tau-o2m 0.5 --tau-no 0.2 --tau-dup 0.7 --mask-thr 0.5 \
  --stability-low 0.40 --stability-high 0.60 --mask-rel-thr 0.744898 --min-mask-pixels 16 \
  --mard-lambda0 0 --mard-lambda-max 0 \
  --device 0 --seed 29 --save-interval 60 \
  > logs/seg/ablation/cvc_maskdhf_only_60ep.log 2>&1

# --- Mask-DHF + DURR (SegMARD forced off) -----------------------------------
mkdir -p runs/seg/dense_sfseg/cvc_maskdhf_durr_60ep logs/seg/ablation
python scripts/YOLO26/medseg/stage2_dense_sfseg_durr.py \
  --weights runs/seg/stage1/frozen/yolo26s_seg_cvc_adabn.pt \
  --target-images dataset/CVC-ClinicDB-YOLO26/images/target \
  --out-dir runs/seg/dense_sfseg/cvc_maskdhf_durr_60ep \
  --imgsz 640 --batch 4 --workers 4 --epochs 60 --lr 1e-4 --device 0 --seed 29 \
  --tau-o2o 0.5 --tau-o2m 0.5 --tau-no 0.2 --tau-dup 0.7 --mask-thr 0.5 \
  --stability-low 0.40 --stability-high 0.60 --mask-rel-thr 0.744898 --min-mask-pixels 16 \
  --durr-tau-match 0.5 --durr-max-witnesses 5 --durr-boundary-kernel 5 --durr-route-gain 1.0 \
  --durr-min-disagreement 0.02 --durr-rescue-conf 0.80 --durr-rescue-stability 0.80 \
  --durr-rescue-consensus-iou 0.70 --durr-rescue-min-support 0 --durr-evidence-conf 0.10 \
  --durr-safe-bg-teacher-prob 0.10 --durr-hall-student-thr 0.80 --durr-hall-area-thr 0.10 \
  --durr-hall-area-weight 0.25 --durr-lambda-dir 0.10 --durr-lambda-rescue 0.20 \
  --durr-lambda-hall 0.10 --durr-warmup-epochs 5 --durr-signed-mode signed_reliability \
  --mard-lambda0 0 --mard-lambda-max 0 \
  --save-interval 60 \
  > logs/seg/ablation/cvc_maskdhf_durr_60ep.log 2>&1

# --- Full MedRT-SFSeg (Mask-DHF + DURR + SegMARD-v2) ------------------------
# Same as README §19's existing K2C command (SegMARD defaults left on:
# --mard-lambda0 0.05 --mard-lambda-max 0.20). Reuse that checkpoint if you
# already have a correctly-labeled one; otherwise rerun it with
# --out-dir runs/seg/dense_sfseg/cvc_full_60ep.
```

Mirror all three for CVC→Kvasir (`--weights
runs/seg/stage1/frozen/yolo26s_seg_kvasir_adabn.pt`, Kvasir target images,
`--mask-rel-thr 0.753071`).

---

## 3. Size-stratified + boundary + hallucination tables (items 1, 2, 3)

One command produces `table0_overall`, `table1_size_stratified` (item 1) and
`table2_hallucination_fp` (item 2's metrics) together, with Boundary
IoU/BF-score included everywhere (item 3):

```bash
python scripts/YOLO26/medseg/evaluate_stratified.py \
  --model baseline=runs/seg/source/frozen/kvasir_yolo26s_seg_source_best.pt \
  --model maskdhf=runs/seg/dense_sfseg/cvc_maskdhf_only_60ep/checkpoints/dense_sfseg_epoch_60.pt \
  --model durr=runs/seg/dense_sfseg/cvc_maskdhf_durr_60ep/checkpoints/durr_student_epoch_60.pt \
  --model full=runs/seg/dense_sfseg/cvc_full_60ep/checkpoints/durr_student_epoch_60.pt \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --out-dir runs/seg/analysis/stratified/k2c \
  --imgsz 640 --conf 0.25 --device 0 \
  --weak-evidence-max-gt-px 0
```

Outputs in `runs/seg/analysis/stratified/k2c/`:
- `table1_size_stratified.md` — item 1's table (Method × Small/Medium/Large × Dice/IoU/ASD/HD95/BF-score/Boundary-IoU). Size thresholds are printed at the top of the run log and saved to `size_bin_thresholds.json` — **report these thresholds in the paper** (e.g. as a footnote) since they're computed from this exact eval set, not fixed constants.
- `table2_hallucination_fp.md` — item 2's table (Precision, FP pixel rate, # FP components, FP area ratio) on the full set and on the `weak_no_evidence` subset (strictly GT-empty images by default). **Compare the `maskdhf` row vs the `durr`/`full` rows on the `weak_no_evidence` subset** — that's the direct evidence for safe-hallucination-suppression.
- `per_image_metrics.csv` — every metric per image per method, for any custom slicing/plotting later.

Repeat with `--model baseline=runs/seg/source/frozen/cvc_yolo26s_seg_source_best.pt` and the three C2K checkpoints, Kvasir images/GT, `--out-dir runs/seg/analysis/stratified/c2k`.

`--weak-evidence-max-gt-px` controls the "no/weak polyp evidence" definition
(default 0 = strictly GT-empty). If you want to also include *barely
visible* polyps in that subset (not just fully absent ones), raise it, e.g.
`--weak-evidence-max-gt-px 50`.

---

## 4. Qualitative O2O → O2M → Δ → final figure (item 2)

Uses the **Full** run's final EMA Teacher + final Student:

```bash
python scripts/YOLO26/medseg/qualitative_o2o_o2m_panel.py \
  --teacher-weights runs/seg/dense_sfseg/cvc_full_60ep/checkpoints/durr_teacher_ema_epoch_60.pt \
  --student-weights runs/seg/dense_sfseg/cvc_full_60ep/checkpoints/durr_student_epoch_60.pt \
  --images dataset/CVC-ClinicDB-YOLO26/images/target \
  --gt-masks dataset/CVC-ClinicDB-YOLO26/gt_masks/target \
  --out-dir runs/seg/analysis/qualitative/k2c_full \
  --num-examples 4 --imgsz 640 --device 0
```

This auto-ranks all target images by mean `|signed Δ|` and picks the 4
strongest-disagreement cases (use `--rank-pool-size 200` to only scan the
first 200 images if the full scan is too slow, or `--image-list
"img1.png,img2.png,img3.png,img4.png"` to hand-pick examples yourself).
Produces one panel PNG per example
(`runs/seg/analysis/qualitative/k2c_full/panels/*.png`, each row = Image |
Teacher O2O | Teacher O2M witness | Signed Δ (diverging colormap) | Final
Student vs GT boundary overlay) plus a stacked `contact_sheet.png` with all
examples — that contact sheet is Fig. 2's replacement/addition for item 2's
qualitative figure. Repeat for the C2K direction.

---

## Notes

- All new scripts are evaluation-only; none of them touch adaptation, so
  none of your existing training logic changes except the additive
  `--durr-signed-mode` flag (defaults to `signed_reliability`, i.e. your
  existing behavior, so `stage2_dense_sfseg_durr.py` runs without that flag
  are byte-identical to before).
- `evaluate_stratified.py` and `boundary_metrics.py` deliberately re-derive
  Dice/IoU/ASD/HD95 locally (not by importing `analyze_seg_results.py`)
  using the *same* formulas/conventions, so the script has no import-order
  dependency on the rest of the codebase and can be dropped anywhere.
- ASD/HD95 here use the same unit-spacing (1.0, 1.0) convention as the rest
  of the repo — see the standing note in `analyze_seg_results.py` about
  Kvasir/CVC not shipping calibrated physical spacing.
