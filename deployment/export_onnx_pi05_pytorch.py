#!/usr/bin/env python3
import argparse
import importlib.util
import logging
import os
from pathlib import Path
import sys

import numpy as np


def _maybe_reexec_in_project_venv() -> None:
    if importlib.util.find_spec("torch") is not None and importlib.util.find_spec("jax") is not None:
        return
    repo_root = Path(__file__).resolve().parents[2]
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]])


_maybe_reexec_in_project_venv()

import torch
from torch.onnx._internal import onnx_proto_utils

import openpi.models_pytorch.pi0_pytorch as pi0_pytorch_mod
from pi05_deploy_utils import INPUT_NAMES
from pi05_deploy_utils import PI05SampleActionsWrapper
from pi05_deploy_utils import configure_runtime
from pi05_deploy_utils import load_pi05_policy
from pi05_deploy_utils import maybe_reexec_in_project_venv
from pi05_deploy_utils import prepare_inputs_from_policy


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export OpenPI pi05 PyTorch model to ONNX")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(Path(__file__).resolve().parent),
        help="Path to converted PyTorch checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "onnx"),
        help="Output directory",
    )
    parser.add_argument("--onnx_name", type=str, default="pi05_sample_actions.onnx")
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--opset", type=int, default=19)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument(
        "--export_dtype",
        type=str,
        default="fp32",
        choices=["fp32"],
        help="Export graph dtype",
    )
    args = parser.parse_args()

    maybe_reexec_in_project_venv()
    configure_runtime()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / args.onnx_name

    logger.info("Loading policy from: %s", args.checkpoint_dir)
    policy, resolved_checkpoint = load_pi05_policy(args.checkpoint_dir)
    policy._model = policy._model.to(device)
    if args.export_dtype == "fp32":
        policy._model = policy._model.float()
    policy._model.eval()

    wrapper = PI05SampleActionsWrapper(policy._model, device=device, num_steps=args.num_steps).to(device)
    wrapper.eval()

    # TensorRT parser compatibility: ensure CumSum runs on int dtype, not bool.
    original_make_att_2d_masks = pi0_pytorch_mod.make_att_2d_masks

    def _make_att_2d_masks_trt_safe(pad_masks, att_masks):
        if att_masks.dtype == torch.bool:
            att_masks = att_masks.to(torch.int32)
        return original_make_att_2d_masks(pad_masks, att_masks)

    pi0_pytorch_mod.make_att_2d_masks = _make_att_2d_masks_trt_safe

    inputs = prepare_inputs_from_policy(policy, device=device, seed=args.seed)
    input_tuple = tuple(inputs[name] for name in INPUT_NAMES)

    with torch.inference_mode():
        ref_out = wrapper(*input_tuple).detach().cpu().numpy()

    dynamic_axes = {
        "state": {0: "batch_size"},
        "base_image": {0: "batch_size"},
        "left_wrist_image": {0: "batch_size"},
        "right_wrist_image": {0: "batch_size"},
        "base_mask": {0: "batch_size"},
        "left_mask": {0: "batch_size"},
        "right_mask": {0: "batch_size"},
        "tokenized_prompt": {0: "batch_size", 1: "prompt_len"},
        "tokenized_prompt_mask": {0: "batch_size", 1: "prompt_len"},
        "noise": {0: "batch_size"},
        "actions": {0: "batch_size"},
    }

    logger.info("Exporting ONNX to: %s", onnx_path)
    # Compatibility shim: some environments have an older/no onnx package.
    # If Torch exporter fails while trying to append ONNXScript functions, bypass that step.
    original_add_onnxscript_fn = onnx_proto_utils._add_onnxscript_fn

    def _safe_add_onnxscript_fn(model_bytes, custom_opsets):
        try:
            return original_add_onnxscript_fn(model_bytes, custom_opsets)
        except Exception as exc:  # pragma: no cover - environment compatibility
            logger.warning("Skipping _add_onnxscript_fn due to exporter environment issue: %s", exc)
            return model_bytes

    onnx_proto_utils._add_onnxscript_fn = _safe_add_onnxscript_fn
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            input_tuple,
            str(onnx_path),
            export_params=True,
            do_constant_folding=False,
            opset_version=args.opset,
            input_names=INPUT_NAMES,
            output_names=["actions"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
            optimize=False,
        )
    pi0_pytorch_mod.make_att_2d_masks = original_make_att_2d_masks
    onnx_proto_utils._add_onnxscript_fn = original_add_onnxscript_fn

    npz_path = out_dir / "sample_io.npz"
    np.savez(
        npz_path,
        **{k: v.detach().cpu().numpy() for k, v in inputs.items()},
        actions_ref=ref_out,
    )
    logger.info("Saved sample I/O: %s", npz_path)
    logger.info("Resolved checkpoint: %s", resolved_checkpoint)
    logger.info("Exported ONNX: %s", onnx_path)


if __name__ == "__main__":
    main()
