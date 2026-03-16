"""Distill OpenPI pi05 with a teacher-student PyTorch workflow.

Example:
    uv run deployment/distill.py \
      --config-name pi05_libero \
      --exp-name pi05_libero_distill \
      --teacher-checkpoint /path/to/teacher_checkpoint \
      --student-init-checkpoint /path/to/student_init_checkpoint \
      --num-train-steps 2000 \
      --batch-size 64 \
      --overwrite

Notes:
    - Teacher/student checkpoints should contain a `model.safetensors` file directly
      or inside numeric step subdirectories.
    - If you only have a JAX checkpoint (`params`), convert it first:
      `uv run examples/convert_jax_model_to_pytorch.py ...`
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import shutil
import time
from typing import Any

import jax
import numpy as np
import safetensors.torch
import torch
import tqdm

import openpi.models.pi0_config as pi0_config
import openpi.models_pytorch.pi0_pytorch as pi0_pytorch
from openpi.shared import download as _download
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class _Formatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = _Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        root.handlers[0].setFormatter(formatter)


def _resolve_checkpoint_root(path_or_url: str) -> pathlib.Path:
    return _download.maybe_download(path_or_url)


def _resolve_model_path(path_or_url: str) -> pathlib.Path:
    resolved = _resolve_checkpoint_root(path_or_url)
    if resolved.is_file():
        if resolved.name == "model.safetensors":
            return resolved
        raise ValueError(f"Expected model.safetensors file, got: {resolved}")

    direct = resolved / "model.safetensors"
    if direct.exists():
        return direct

    step_dirs = [
        d for d in resolved.iterdir() if d.is_dir() and d.name.isdigit() and (d / "model.safetensors").exists()
    ]
    if step_dirs:
        latest = max(step_dirs, key=lambda d: int(d.name))
        return latest / "model.safetensors"

    if (resolved / "params").exists():
        raise ValueError(
            f"Found JAX checkpoint at {resolved}, but no PyTorch model.safetensors. "
            "Convert first with `uv run examples/convert_jax_model_to_pytorch.py`."
        )

    recursive = sorted(resolved.rglob("model.safetensors"))
    if recursive:
        return max(recursive, key=lambda p: p.stat().st_mtime)

    raise FileNotFoundError(f"Could not find model.safetensors under: {resolved}")


def _to_pi0_config(
    model_cfg: Any,
    *,
    precision: str,
    paligemma_variant: str | None = None,
    action_expert_variant: str | None = None,
) -> pi0_config.Pi0Config:
    if isinstance(model_cfg, pi0_config.Pi0Config):
        cfg = dataclasses.replace(model_cfg)
    else:
        cfg = pi0_config.Pi0Config(
            action_dim=model_cfg.action_dim,
            action_horizon=model_cfg.action_horizon,
            max_token_len=model_cfg.max_token_len,
            paligemma_variant=getattr(model_cfg, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(model_cfg, "action_expert_variant", "gemma_300m"),
            pi05=getattr(model_cfg, "pi05", False),
            discrete_state_input=getattr(model_cfg, "discrete_state_input", None),
        )

    cfg = dataclasses.replace(
        cfg,
        dtype=precision,
        paligemma_variant=paligemma_variant or cfg.paligemma_variant,
        action_expert_variant=action_expert_variant or cfg.action_expert_variant,
    )
    return cfg


def _lr_for_step(step: int, warmup_steps: int, peak_lr: float, decay_steps: int, end_lr: float) -> float:
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    cos = 0.5 * (1.0 + np.cos(np.pi * progress))
    return end_lr + (peak_lr - end_lr) * cos


def _move_observation_to_device(observation: Any, device: torch.device) -> Any:
    observation = jax.tree.map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, observation)

    # Some sources (for example fake/debug data) produce images in NHWC.
    # PI0Pytorch expects NCHW at this stage.
    converted_images = {}
    changed = False
    for key, image in observation.images.items():
        if image.ndim == 4 and image.shape[1] != 3 and image.shape[-1] == 3:
            converted_images[key] = image.permute(0, 3, 1, 2).contiguous()
            changed = True
        else:
            converted_images[key] = image

    if changed:
        observation = dataclasses.replace(observation, images=converted_images)
    return observation


def _save_checkpoint(
    *,
    output_dir: pathlib.Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    data_config: _config.DataConfig,
) -> None:
    final_dir = output_dir / f"{step}"
    tmp_dir = output_dir / f"tmp_{step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safetensors.torch.save_model(model, tmp_dir / "model.safetensors")
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(metadata, tmp_dir / "metadata.pt")

    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(tmp_dir / "assets" / data_config.asset_id, data_config.norm_stats)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)


def _build_model(
    model_cfg: pi0_config.Pi0Config,
    device: torch.device,
    *,
    role: str,
    allow_cpu_fallback: bool = True,
) -> tuple[pi0_pytorch.PI0Pytorch, torch.device]:
    try:
        model = pi0_pytorch.PI0Pytorch(model_cfg).to(device)
        return model, device
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        if allow_cpu_fallback and is_oom and device.type == "cuda":
            logging.warning(f"{role} model OOM on {device}. Falling back to CPU.")
            torch.cuda.empty_cache()
            cpu = torch.device("cpu")
            model = pi0_pytorch.PI0Pytorch(model_cfg).to(cpu)
            return model, cpu
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill OpenPI pi05 in PyTorch.")
    parser.add_argument("--config-name", type=str, default="pi05_libero")
    parser.add_argument("--exp-name", type=str, default="distill_openpi05")
    parser.add_argument("--checkpoint-base-dir", type=str, default="./checkpoints")
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--student-init-checkpoint", type=str, default=None)
    parser.add_argument("--allow-random-teacher", action="store_true")
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--precision", type=str, choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--teacher-num-steps", type=int, default=4)
    parser.add_argument("--gt-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--decay-steps", type=int, default=None)
    parser.add_argument("--end-lr", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--teacher-device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable-compile", action="store_true")
    parser.add_argument("--student-paligemma-variant", type=str, default=None)
    parser.add_argument("--student-action-expert-variant", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    init_logging()
    args = _parse_args()
    if not args.enable_compile:
        os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    base_cfg = _config.get_config(args.config_name)
    distill_cfg = dataclasses.replace(
        base_cfg,
        exp_name=args.exp_name,
        checkpoint_base_dir=args.checkpoint_base_dir,
        overwrite=args.overwrite,
        num_train_steps=args.num_train_steps if args.num_train_steps is not None else base_cfg.num_train_steps,
        batch_size=args.batch_size if args.batch_size is not None else base_cfg.batch_size,
        num_workers=args.num_workers if args.num_workers is not None else base_cfg.num_workers,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        pytorch_training_precision=args.precision,
        wandb_enabled=False,
    )

    if not isinstance(distill_cfg.model, pi0_config.Pi0Config) or not distill_cfg.model.pi05:
        raise ValueError(
            f"Config '{args.config_name}' is not a pi05 config. "
            "Use a pi05 config (for example: pi05_libero, pi05_droid, debug_pi05)."
        )

    out_dir = distill_cfg.checkpoint_dir
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is non-empty: {out_dir}. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Writing distillation checkpoints to: {out_dir}")

    loader = _data.create_data_loader(distill_cfg, framework="pytorch", shuffle=True)
    data_config = loader.data_config()

    student_device = (
        torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    teacher_device = torch.device(args.teacher_device) if args.teacher_device else student_device
    logging.info(f"Using devices: student={student_device}, teacher={teacher_device}")

    teacher_cfg = _to_pi0_config(distill_cfg.model, precision=args.precision)
    student_cfg = _to_pi0_config(
        distill_cfg.model,
        precision=args.precision,
        paligemma_variant=args.student_paligemma_variant,
        action_expert_variant=args.student_action_expert_variant,
    )

    teacher, teacher_device = _build_model(teacher_cfg, teacher_device, role="teacher")
    teacher = teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    teacher_source = args.teacher_checkpoint
    if teacher_source is None and distill_cfg.pytorch_weight_path:
        candidate = pathlib.Path(distill_cfg.pytorch_weight_path)
        if candidate.exists():
            teacher_source = str(candidate)

    if teacher_source:
        teacher_model_path = _resolve_model_path(teacher_source)
        safetensors.torch.load_model(teacher, teacher_model_path, device=str(teacher_device), strict=True)
        logging.info(f"Loaded teacher model from: {teacher_model_path}")
    elif not args.allow_random_teacher:
        raise ValueError("No --teacher-checkpoint provided. Pass one, or set --allow-random-teacher for debugging.")
    else:
        logging.warning("Using randomly initialized teacher (debug only).")

    student, student_device = _build_model(student_cfg, student_device, role="student")
    student = student.train()

    if args.student_init_checkpoint:
        student_model_path = _resolve_model_path(args.student_init_checkpoint)
        safetensors.torch.load_model(student, student_model_path, device=str(student_device), strict=False)
        logging.info(f"Initialized student from: {student_model_path}")
    elif distill_cfg.pytorch_weight_path and pathlib.Path(distill_cfg.pytorch_weight_path).exists():
        student_model_path = _resolve_model_path(distill_cfg.pytorch_weight_path)
        safetensors.torch.load_model(student, student_model_path, device=str(student_device), strict=False)
        logging.info(f"Initialized student from config pytorch_weight_path: {student_model_path}")
    else:
        logging.warning("Student starts from random initialization.")

    peak_lr = args.lr if args.lr is not None else distill_cfg.lr_schedule.peak_lr
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else distill_cfg.lr_schedule.warmup_steps
    decay_steps = args.decay_steps if args.decay_steps is not None else distill_cfg.lr_schedule.decay_steps
    end_lr = args.end_lr if args.end_lr is not None else distill_cfg.lr_schedule.decay_lr

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=peak_lr,
        betas=(distill_cfg.optimizer.b1, distill_cfg.optimizer.b2),
        eps=distill_cfg.optimizer.eps,
        weight_decay=distill_cfg.optimizer.weight_decay,
    )

    total_steps = distill_cfg.num_train_steps
    logging.info(
        "Distillation setup: "
        f"steps={total_steps}, batch_size={distill_cfg.batch_size}, "
        f"teacher_num_steps={args.teacher_num_steps}, gt_loss_weight={args.gt_loss_weight}"
    )

    loader_iter = iter(loader)
    recent = []
    pbar = tqdm.tqdm(range(total_steps), desc="Distill", total=total_steps)
    start_time = time.time()

    for step in pbar:
        raw_observation, raw_actions = next(loader_iter)

        lr = _lr_for_step(step, warmup_steps, peak_lr, decay_steps, end_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        retried_teacher_on_cpu = False
        while True:
            try:
                teacher_observation = _move_observation_to_device(raw_observation, teacher_device)
                with torch.inference_mode():
                    teacher_actions = teacher.sample_actions(
                        device=teacher_device,
                        observation=teacher_observation,
                        num_steps=args.teacher_num_steps,
                    )
                break
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower()
                if is_oom and teacher_device.type == "cuda" and not retried_teacher_on_cpu:
                    logging.warning("Teacher OOM on CUDA during sampling. Moving teacher to CPU and retrying step.")
                    teacher = teacher.to("cpu")
                    teacher_device = torch.device("cpu")
                    torch.cuda.empty_cache()
                    retried_teacher_on_cpu = True
                    continue
                if is_oom:
                    if student_device.type == "cuda":
                        torch.cuda.empty_cache()
                    raise RuntimeError(
                        f"OOM at step {step + 1}. Try lower --batch-size, lower --teacher-num-steps, "
                        "or set --teacher-device cpu."
                    ) from exc
                raise

        try:
            observation = _move_observation_to_device(raw_observation, student_device)
            actions = raw_actions.to(device=student_device, dtype=torch.float32)
            teacher_actions = teacher_actions.to(device=student_device, dtype=torch.float32)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                if student_device.type == "cuda":
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"OOM at step {step + 1}. Try lower --batch-size or use --device cpu."
                ) from exc
            raise

        kd_loss = student(observation, teacher_actions).mean()
        if args.gt_loss_weight > 0:
            gt_loss = student(observation, actions).mean()
            loss = kd_loss + args.gt_loss_weight * gt_loss
        else:
            gt_loss = None
            loss = kd_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
        optimizer.step()

        step_stats = {
            "loss": float(loss.item()),
            "kd_loss": float(kd_loss.item()),
            "gt_loss": float(gt_loss.item()) if gt_loss is not None else 0.0,
            "lr": float(lr),
            "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
        }
        recent.append(step_stats)
        pbar.set_postfix(loss=f"{step_stats['loss']:.4f}", lr=f"{step_stats['lr']:.2e}")

        save_now = ((step + 1) % distill_cfg.save_interval == 0) or (step == total_steps - 1)
        if save_now:
            metadata = {
                "step": step + 1,
                "args": vars(args),
                "train_config": dataclasses.asdict(distill_cfg),
                "teacher_checkpoint": teacher_source,
                "student_init_checkpoint": args.student_init_checkpoint,
                "timestamp": time.time(),
            }
            _save_checkpoint(
                output_dir=out_dir,
                step=step + 1,
                model=student,
                optimizer=optimizer,
                metadata=metadata,
                data_config=data_config,
            )
            logging.info(f"Saved checkpoint at step {step + 1}")

        if (step + 1) % distill_cfg.log_interval == 0:
            elapsed = time.time() - start_time
            avg = {k: sum(d[k] for d in recent) / len(recent) for k in recent[0]}
            logging.info(
                f"step={step + 1} loss={avg['loss']:.4f} kd={avg['kd_loss']:.4f} "
                f"gt={avg['gt_loss']:.4f} grad_norm={avg['grad_norm']:.3f} "
                f"lr={avg['lr']:.2e} time={elapsed:.1f}s"
            )
            start_time = time.time()
            recent = []

    logging.info(f"Distillation finished. Final checkpoint directory: {out_dir}")


if __name__ == "__main__":
    main()
