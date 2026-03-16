# OpenPI pi05 Distillation Pipeline Summary

This document summarizes the `deployment/distill.py` pipeline for distilling a `pi05` student from a `pi05` teacher.

## Purpose

- Train a student policy with teacher-generated action targets (`knowledge distillation`).
- Reuse OpenPI training configs, dataloaders, model stack, and checkpoint format.
- Support practical runtime constraints (shape compatibility, OOM fallback, optional compile).

## Entry Point

- Script: `deployment/distill.py`
- Main command pattern:

```bash
uv run deployment/distill.py \
  --config-name pi05_libero \
  --exp-name pi05_libero_distill \
  --teacher-checkpoint /path/to/teacher_ckpt \
  --student-init-checkpoint /path/to/student_init_ckpt \
  --num-train-steps 2000 \
  --batch-size 64 \
  --device cuda:1 \
  --teacher-device cpu \
  --overwrite
```

## High-Level Flow

1. Parse CLI args and load `TrainConfig` by `--config-name`.
2. Build a distill config override (`exp_name`, steps, batch size, save/log intervals, precision).
3. Create data loader via `openpi.training.data_loader.create_data_loader(..., framework="pytorch")`.
4. Build teacher and student `PI0Pytorch` models from the config’s `Pi0Config`.
5. Resolve and load checkpoints (`model.safetensors`) for teacher/student.
6. For each step:
   - Get batch from dataset.
   - Run teacher `sample_actions(...)` for distillation targets.
   - Compute student KD loss: `student(observation, teacher_actions).mean()`.
   - Optional GT loss blend if `--gt-loss-weight > 0`.
   - Backprop + grad clipping + optimizer step.
7. Save checkpoints at intervals and final step.

## Inputs

- `--config-name`: must map to a `pi05` config.
- Teacher checkpoint:
  - `--teacher-checkpoint` or fallback to config `pytorch_weight_path` if present.
  - Accepts local path or remote path resolvable by OpenPI download utilities.
- Student init checkpoint (optional):
  - `--student-init-checkpoint` or fallback to config `pytorch_weight_path`.
- Data assets:
  - Norm stats expected via config assets path (for `pi05_libero`, typically `assets/pi05_libero/...`).

## Checkpoint Resolution Rules

When a checkpoint path is provided, resolver tries:

1. Direct file `model.safetensors`
2. `<path>/model.safetensors`
3. Latest numeric step dir `<path>/<step>/model.safetensors`
4. Recursive search under path

If only JAX `params` exists, conversion is required first:

```bash
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /path/to/jax_ckpt \
  --config_name pi05_libero \
  --output_path /path/to/pytorch_ckpt
```

## Output Structure

Outputs are written to:

- `checkpoints/<config_name>/<exp_name>/<step>/`

Each saved step contains:

- `model.safetensors`
- `optimizer.pt`
- `metadata.pt`
- `assets/<asset_id>/norm_stats.json` (when available)

## Stability and Reliability Features

- **Compile safety**:
  - `sample_actions` compile is optional.
  - Distill defaults to eager mode (`OPENPI_DISABLE_TORCH_COMPILE=1`) unless `--enable-compile`.
- **Shape safety**:
  - Observation images are normalized to expected layout (NCHW) before model calls.
  - Vision projection dimension in `gemma_pytorch.py` is aligned with model width to avoid dummy/debug mismatches.
- **OOM handling**:
  - Teacher/student can run on different devices (`--teacher-device`, `--device`).
  - Teacher build supports CUDA->CPU fallback on OOM.
  - Per-step teacher sampling retries on CPU if teacher CUDA OOM occurs.
  - Clear error guidance suggests lowering batch size/teacher steps/device changes.

## Loss Definition

- KD loss:
  - `kd_loss = student(observation, teacher_actions).mean()`
- Optional supervised blend:
  - `loss = kd_loss + gt_loss_weight * student(observation, actions).mean()`

## Key Tunables

- `--num-train-steps`
- `--batch-size`
- `--teacher-num-steps`
- `--gt-loss-weight`
- `--lr`, `--warmup-steps`, `--decay-steps`, `--end-lr`
- `--max-grad-norm`
- `--device`, `--teacher-device`
- `--enable-compile`

## Practical Recommendations

- Start with a smoke run:
  - `--num-train-steps 1 --batch-size 1 --teacher-num-steps 1`
- If GPU memory is tight:
  - put teacher on CPU (`--teacher-device cpu`)
  - reduce `--batch-size`
  - reduce `--teacher-num-steps`
- Ensure transformers patch is applied in the env for OpenPI PyTorch models.

