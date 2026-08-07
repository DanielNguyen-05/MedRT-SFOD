"""Stage-2 RASP-SFOD for YOLO26 end-to-end detectors.

This script is a strict research extension of ``stage2_rtsfod_yolo26.py``:
Mean-Teacher, DHF, MARD, augmentation, native YOLO26 O2O/O2M loss and epoch-level
EMA are retained.  The only new learning mechanism is student-only adaptive
structured pruning from ``rasp_pruning.py``.

Main protocol:
  - Teacher: dense/full-precision, weak target view, DHF pseudo-label source.
  - Student: same YOLO26 architecture and dense latent parameters, strong view.
  - RASP: target Taylor importance + GMM + cost-aware ranking + knee budget.
  - Accepted masks are monotonic and updated only at epoch boundaries.
  - Continued RT-SFOD + MARD between pruning events provides implicit recovery.
  - Physical channel removal is a separate export step after adaptation.

Do not enable ``--eval`` for main source-free model selection: target labels are
allowed only for final reporting / diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
import torch.utils.data as data

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from ultralytics import YOLO  # noqa: E402

from rasp_pruning import RASPController, add_rasp_args, save_audit_json  # noqa: E402
from stage2_rtsfod_yolo26 import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_GRAD_CLIP,
    DEFAULT_IMGSZ,
    DEFAULT_LR,
    EMA_MOMENTUM,
    MARD_ALPHA,
    MARD_BETA,
    MARD_BG_POINTS,
    MARD_ETA,
    MARD_FG_POINTS,
    MARD_GATE_THRESHOLD,
    MARD_GAMMA,
    MARD_INTERVAL,
    MARD_LAMBDA0,
    MARD_LAMBDA_MAX,
    MARD_TOPK_BOXES,
    MARD_WARMUP_EPOCHS,
    TAU_DUP,
    TAU_NO,
    TAU_O2M,
    TAU_O2O,
    DetectInputFeatureHook,
    TargetTeacherStudentDataset,
    average_pseudo_confidence,
    collate_fn,
    compute_mard_loss,
    compute_student_loss,
    ensure_detection_loss_args,
    generate_pseudo_labels,
    list_images_from_yaml,
    map_pseudo_labels_to_strong,
    mard_weight,
    resolve_device,
    scalarize,
    seed_everything,
    seed_worker,
    setup_teacher_student,
    update_teacher_ema,
    val_device_arg,
)


def _state_path(out_dir: Path, epoch: int) -> Path:
    return out_dir / "checkpoints" / f"rasp_training_state_epoch_{epoch}.pt"


def _latest_state_path(out_dir: Path) -> Path:
    return out_dir / "checkpoints" / "rasp_training_state_latest.pt"


def _save_training_state(
    path: Path,
    *,
    epoch: int,
    global_step: int,
    teacher_model,
    student_model,
    optimizer,
    scheduler,
    criterion,
    rasp: RASPController,
    args,
) -> None:
    payload = {
        "format": "RASP-SFOD-v1",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "teacher_state": teacher_model.state_dict(),
        "student_state": student_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "criterion_state": criterion.state_dict() if hasattr(criterion, "state_dict") else None,
        "rasp": rasp.state_dict(),
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_training_state(
    path: str,
    *,
    teacher_model,
    student_model,
    optimizer,
    scheduler,
    criterion,
    rasp: RASPController,
    device: torch.device,
) -> tuple[int, int]:
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") != "RASP-SFOD-v1":
        warnings.warn(f"Unknown RASP state format: {state.get('format')}")
    teacher_model.load_state_dict(state["teacher_state"], strict=True)
    student_model.load_state_dict(state["student_state"], strict=True)
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    if state.get("criterion_state") is not None and hasattr(criterion, "load_state_dict"):
        criterion.load_state_dict(state["criterion_state"])
    rasp.load_state_dict(state.get("rasp", {"enabled": False}), strict=True)
    rng = state.get("rng", {})
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception as exc:
            warnings.warn(f"Could not restore CUDA RNG states: {exc}")
    return int(state.get("epoch", 0)), int(state.get("global_step", 0))


def _write_history(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(args: argparse.Namespace) -> None:
    if args.eval:
        warnings.warn(
            "--eval reads target validation labels. Keep it OFF for the main source-free training protocol; "
            "use only for diagnostics/final reporting."
        )
    seed_everything(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "rasp_history.jsonl"

    imgs = list_images_from_yaml(args.data)
    assert imgs, f"No target training images found in {args.data}"
    print(
        f"[RASP Stage 2] YOLO26 | device={device} images={len(imgs)} imgsz={args.imgsz} "
        f"batch={args.batch} epochs={args.epochs} rasp={args.rasp_enable}",
        flush=True,
    )
    print(
        f"[RASP] DHF tau_o2o={args.tau_o2o} tau_o2m={args.tau_o2m} "
        f"tau_no={args.tau_no} tau_dup={args.tau_dup}",
        flush=True,
    )
    print(
        f"[RASP] MARD lambda0={args.mard_lambda0} gamma={args.mard_gamma} "
        f"alpha={args.mard_alpha} beta={args.mard_beta} warmup={args.mard_warmup_epochs}ep",
        flush=True,
    )
    if args.rasp_enable:
        print(
            f"[RASP] adaptive pruning warmup={args.rasp_warmup_epochs}ep cycle={args.rasp_cycle_epochs}ep "
            f"reliability>={args.rasp_reliability_threshold} keep>={args.rasp_min_keep_ratio:.2f} "
            f"round_to={args.rasp_round_to}",
            flush=True,
        )

    dataset = TargetTeacherStudentDataset(imgs, img_size=args.imgsz)
    generator = None
    if args.seed is not None and args.seed >= 0:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    dataloader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=args.workers > 0,
    )

    teacher_model, student_model, student_wrapper = setup_teacher_student(args.stage1_model, device)
    ensure_detection_loss_args(student_model, args.epochs)
    student_model.criterion = student_model.init_criterion()
    criterion = student_model.criterion

    optimizer = optim.SGD(
        student_model.parameters(),
        lr=args.lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    rasp = RASPController(student_model, args)
    hook: Optional[DetectInputFeatureHook] = DetectInputFeatureHook(student_model)

    # Optional one-time DepGraph audit.  It is deliberately separate from the
    # masking logic; failure means the current YOLO fork should be inspected,
    # not that the training loop should silently prune a different structure.
    if args.rasp_enable and args.rasp_depgraph_audit:
        audit_size = int(args.rasp_audit_imgsz)
        example = torch.randn(1, 3, audit_size, audit_size, device=device)
        print(f"[RASP] running DepGraph audit with {audit_size}x{audit_size} dummy input...", flush=True)
        audit = rasp.pruner.audit_depgraph(example)
        save_audit_json(out_dir / "rasp_depgraph_audit.json", audit)
        bad = [name for name, result in audit.items() if not result["ok"]]
        print(f"[RASP] DepGraph audit: {len(audit)-len(bad)}/{len(audit)} local-safe groups", flush=True)
        if bad:
            print(f"[RASP] audit warnings, first groups: {bad[:5]}", flush=True)
            if args.rasp_require_depgraph:
                raise RuntimeError(f"DepGraph audit failed for {len(bad)} groups; see rasp_depgraph_audit.json")

    start_epoch = 0
    global_step = 0
    if args.resume_state:
        start_epoch, global_step = _load_training_state(
            args.resume_state,
            teacher_model=teacher_model,
            student_model=student_model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            rasp=rasp,
            device=device,
        )
        print(f"[RASP] resumed from epoch={start_epoch} global_step={global_step}", flush=True)

    try:
        for epoch_idx in range(start_epoch, args.epochs):
            epoch = epoch_idx + 1
            student_model.train()
            rasp.begin_epoch()
            epoch_start = time.time()
            epoch_batches = 0
            skipped_batches = 0
            pseudo_boxes = 0
            conf_sum = 0.0
            conf_count = 0
            sums = {"loss": 0.0, "det": 0.0, "mard": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}

            print(f"[Epoch {epoch:03d}/{args.epochs:03d}] start", flush=True)
            for batch_i, (weak_imgs, strong_imgs, _paths, weak_infos, strong_infos) in enumerate(dataloader, start=1):
                weak_imgs = weak_imgs.to(device, non_blocking=True)
                strong_imgs = strong_imgs.to(device, non_blocking=True)

                pseudo_weak = generate_pseudo_labels(
                    teacher_model,
                    weak_imgs,
                    tau_o2o=args.tau_o2o,
                    tau_o2m=args.tau_o2m,
                    tau_no=args.tau_no,
                    tau_dup=args.tau_dup,
                )
                pseudo_strong = map_pseudo_labels_to_strong(pseudo_weak, weak_infos, strong_infos)
                valid = [labels.numel() > 0 for labels in pseudo_strong]

                if not any(valid):
                    skipped_batches += 1
                    global_step += 1
                    if args.print_freq > 0 and (batch_i == 1 or batch_i % args.print_freq == 0):
                        print(
                            f"[Epoch {epoch:03d}/{args.epochs:03d}] batch {batch_i:04d}/{len(dataloader):04d} "
                            "skipped=no_pseudo_labels",
                            flush=True,
                        )
                    continue

                strong_valid = strong_imgs[valid]
                labels_valid = [labels for labels, keep in zip(pseudo_strong, valid) if keep]
                infos_valid = [info for info, keep in zip(strong_infos, valid) if keep]
                avg_conf = average_pseudo_confidence(labels_valid)
                batch_boxes = sum(labels.shape[0] for labels in labels_valid)
                conf_sum += float(avg_conf) * max(batch_boxes, 1)
                conf_count += max(batch_boxes, 1)

                if hook is not None:
                    hook.latest = None
                student_outputs = student_model(strong_valid)
                feats = hook.latest if hook is not None else None
                loss_dict, det_loss = compute_student_loss(
                    student_outputs=student_outputs,
                    pseudo_labels=labels_valid,
                    student_model=student_model,
                    criterion=criterion,
                    input_shape=strong_valid.shape,
                )

                reg_loss = det_loss.new_zeros(())
                lambda_mard = 0.0
                if feats is not None and global_step % MARD_INTERVAL == 0:
                    reg_loss, _ = compute_mard_loss(
                        feats=feats,
                        pseudo_labels=labels_valid,
                        strong_infos=infos_valid,
                        h_pad=int(strong_valid.shape[2]),
                        w_pad=int(strong_valid.shape[3]),
                        args=args,
                    )
                    lambda_mard = mard_weight(args, global_step, len(dataloader), avg_conf)

                total_loss = det_loss + lambda_mard * reg_loss
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()  # RASP Taylor hooks accumulate here.
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.grad_clip)
                optimizer.step()

                global_step += 1
                epoch_batches += 1
                pseudo_boxes += batch_boxes

                values = {
                    "loss": scalarize(total_loss),
                    "det": scalarize(det_loss),
                    "mard": scalarize(reg_loss),
                    "box": scalarize(loss_dict["box_loss"]),
                    "cls": scalarize(loss_dict["cls_loss"]),
                    "dfl": scalarize(loss_dict["dfl_loss"]),
                }
                for key, value in values.items():
                    sums[key] += value

                if args.print_freq > 0 and (
                    batch_i == 1 or batch_i == len(dataloader) or batch_i % args.print_freq == 0
                ):
                    sparsity = rasp.pruner.sparsity() if args.rasp_enable else 0.0
                    print(
                        f"[Epoch {epoch:03d}/{args.epochs:03d}] batch {batch_i:04d}/{len(dataloader):04d} "
                        f"loss={values['loss']:.4f} det={values['det']:.4f} box={values['box']:.4f} "
                        f"cls={values['cls']:.4f} dfl={values['dfl']:.4f} mard={values['mard']:.4f} "
                        f"lambda={lambda_mard:.4f} pseudo_boxes={batch_boxes} avg_conf={avg_conf:.4f} "
                        f"rasp_s={sparsity:.3f} lr={optimizer.param_groups[0]['lr']:.6g}",
                        flush=True,
                    )

            scheduler.step()
            if hasattr(criterion, "update"):
                criterion.update()

            # Keep the paper's epoch-level EMA semantics.  RASP gates do not
            # modify stored student parameters, so the dense teacher receives
            # dense latent student weights exactly as intended.
            update_teacher_ema(teacher_model, student_model, momentum=args.ema_momentum)

            epoch_reliability = float(conf_sum / max(conf_count, 1))
            prune_stats = rasp.end_epoch(epoch, epoch_reliability)

            denom = max(epoch_batches, 1)
            row = {
                "epoch": epoch,
                "time_sec": time.time() - epoch_start,
                "valid_batches": epoch_batches,
                "skipped_batches": skipped_batches,
                "pseudo_boxes": pseudo_boxes,
                "avg_dhf_conf": epoch_reliability,
                "lr": optimizer.param_groups[0]["lr"],
                **{k: sums[k] / denom for k in sums},
                **prune_stats,
            }
            _write_history(history_path, row)
            print(
                f"[Epoch {epoch:03d}/{args.epochs:03d}] done time={row['time_sec']:.1f}s "
                f"valid={epoch_batches} skipped={skipped_batches} pseudo_boxes={pseudo_boxes} "
                f"loss={row['loss']:.4f} det={row['det']:.4f} mard={row['mard']:.4f} "
                f"conf={epoch_reliability:.4f} "
                + (
                    f"RASP(status={prune_stats.get('rasp_status','-')}, "
                    f"s={prune_stats.get('rasp_hidden_sparsity',0):.3f}, "
                    f"new_packs={prune_stats.get('rasp_new_packs',0)}, "
                    f"knee={prune_stats.get('rasp_knee_packs',0)}/{prune_stats.get('rasp_candidates',0)})"
                    if args.rasp_enable
                    else ""
                ),
                flush=True,
            )

            save_this_epoch = epoch % args.save_interval == 0 or epoch == args.epochs
            eval_this_epoch = args.eval and epoch % args.val_interval == 0

            if save_this_epoch:
                # Save a standard Ultralytics checkpoint of the *dense latent*
                # student plus a complete RASP training state carrying masks.
                latent_path = checkpoint_dir / f"yolo26_rasp_latent_epoch_{epoch}.pt"
                with rasp.suspended():
                    student_wrapper.model = student_model
                    student_wrapper.save(str(latent_path))
                state_path = _state_path(out_dir, epoch)
                _save_training_state(
                    state_path,
                    epoch=epoch,
                    global_step=global_step,
                    teacher_model=teacher_model,
                    student_model=student_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    rasp=rasp,
                    args=args,
                )
                # latest is convenient for Colab preemption/resume.
                _save_training_state(
                    _latest_state_path(out_dir),
                    epoch=epoch,
                    global_step=global_step,
                    teacher_model=teacher_model,
                    student_model=student_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    rasp=rasp,
                    args=args,
                )
                print(f"[Epoch {epoch:03d}] saved latent={latent_path.name} state={state_path.name}", flush=True)

            if eval_this_epoch:
                student_model.eval()
                val_wrapper = YOLO(args.stage1_model)
                val_wrapper.model = student_model
                metrics = val_wrapper.val(
                    data=args.data,
                    imgsz=args.imgsz,
                    batch=args.batch,
                    conf=0.001,
                    iou=0.6,
                    device=val_device_arg(device),
                    plots=False,
                    verbose=False,
                )
                map50 = getattr(metrics.box, "map50", 0.0)
                print(f"[ORACLE diagnostic] epoch={epoch} masked-student mAP50={map50:.4f}", flush=True)
                student_model.train()
    finally:
        if hook is not None:
            hook.close()
        if args.rasp_enable and rasp.pruner is not None:
            rasp.pruner.uninstall()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RASP-SFOD Stage 2 for YOLO26 end-to-end detectors")
    parser.add_argument("--stage1_model", type=str, required=True, help="AdaBN-initialized Stage-1 checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Target-domain data YAML")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--resume_state", type=str, default="", help="Resume from rasp_training_state_*.pt")

    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--grad_clip", type=float, default=DEFAULT_GRAD_CLIP)
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--tau_o2o", type=float, default=TAU_O2O)
    parser.add_argument("--tau_o2m", type=float, default=TAU_O2M)
    parser.add_argument("--tau_no", type=float, default=TAU_NO)
    parser.add_argument("--tau_dup", type=float, default=TAU_DUP)

    parser.add_argument("--mard_lambda0", type=float, default=MARD_LAMBDA0)
    parser.add_argument("--mard_lambda_max", type=float, default=MARD_LAMBDA_MAX)
    parser.add_argument("--mard_gamma", type=float, default=MARD_GAMMA)
    parser.add_argument("--mard_alpha", type=float, default=MARD_ALPHA)
    parser.add_argument("--mard_beta", type=float, default=MARD_BETA)
    parser.add_argument("--mard_warmup_epochs", type=float, default=MARD_WARMUP_EPOCHS)
    parser.add_argument("--mard_gate_threshold", type=float, default=MARD_GATE_THRESHOLD)
    parser.add_argument("--mard_topk_boxes", type=int, default=MARD_TOPK_BOXES)
    parser.add_argument("--mard_fg_points", type=int, default=MARD_FG_POINTS)
    parser.add_argument("--mard_bg_points", type=int, default=MARD_BG_POINTS)
    parser.add_argument("--mard_eta", type=float, default=MARD_ETA)

    parser.add_argument("--ema_momentum", type=float, default=EMA_MOMENTUM)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--eval", action="store_true", help="ORACLE diagnostic only: evaluates target labels")
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=29, help="Use -1 to disable seeding")
    parser.add_argument("--deterministic", action="store_true")
    add_rasp_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
