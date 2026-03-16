#!/usr/bin/env python3
import argparse
import importlib.util
import logging
import os
import time
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

from pi05_deploy_utils import INPUT_NAMES
from pi05_deploy_utils import PI05SampleActionsWrapper
from pi05_deploy_utils import configure_runtime
from pi05_deploy_utils import load_pi05_policy
from pi05_deploy_utils import maybe_reexec_in_project_venv
from pi05_deploy_utils import prepare_inputs_from_policy


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def tensor_metrics(ref: torch.Tensor, pred: torch.Tensor) -> dict[str, float]:
    ref_f = ref.detach().float().reshape(-1).cpu()
    pred_f = pred.detach().float().reshape(-1).cpu()
    diff = pred_f - ref_f
    mae = diff.abs().mean().item()
    max_abs = diff.abs().max().item()
    rel_l2 = torch.linalg.vector_norm(diff).item() / (torch.linalg.vector_norm(ref_f).item() + 1e-12)
    cos = torch.nn.functional.cosine_similarity(ref_f, pred_f, dim=0).item()
    return {"mae": mae, "max_abs": max_abs, "rel_l2": rel_l2, "cosine": cos}


def benchmark_torch(wrapper, inputs, warmup: int, iters: int):
    inp = tuple(inputs[k] for k in INPUT_NAMES)
    for _ in range(warmup):
        with torch.inference_mode():
            _ = wrapper(*inp)
    if inp[0].is_cuda:
        torch.cuda.synchronize()
    times = []
    out = None
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = wrapper(*inp)
        if inp[0].is_cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return out, np.array(times)


def benchmark_onnxruntime(onnx_path: str, inputs, warmup: int, iters: int):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if torch.cuda.is_available():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(onnx_path, providers=providers)
    ort_inputs = {k: inputs[k].detach().cpu().numpy() for k in INPUT_NAMES}

    for _ in range(warmup):
        _ = session.run(["actions"], ort_inputs)
    times = []
    out = None
    for _ in range(iters):
        t0 = time.perf_counter()
        out = session.run(["actions"], ort_inputs)[0]
        times.append((time.perf_counter() - t0) * 1000)
    return torch.from_numpy(out), np.array(times), providers[0]


class TensorRTWrapper:
    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorRT Python module is not available in current interpreter. "
                "Run benchmark from the environment where TensorRT is installed."
            ) from exc

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.context = self.engine.create_execution_context()

    def __call__(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        in_cuda = {k: v.cuda().contiguous() for k, v in inputs.items()}
        for name, t in in_cuda.items():
            self.context.set_input_shape(name, tuple(t.shape))
            self.context.set_tensor_address(name, t.data_ptr())

        out_shape = tuple(self.context.get_tensor_shape("actions"))
        out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        self.context.set_tensor_address("actions", out.data_ptr())

        ok = self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT inference failed")
        return out


def benchmark_trt(engine_path: str, inputs, warmup: int, iters: int):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for TensorRT benchmark")
    trt_runner = TensorRTWrapper(engine_path)
    for _ in range(warmup):
        _ = trt_runner(inputs)
    torch.cuda.synchronize()
    times = []
    out = None
    for _ in range(iters):
        t0 = time.perf_counter()
        out = trt_runner(inputs)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return out, np.array(times)


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenPI pi05: PyTorch vs ONNX Runtime vs TensorRT")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=str(Path(__file__).resolve().parent), help="PyTorch checkpoint dir"
    )
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--trt_engine", type=str, default=None)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--strict_backends",
        action="store_true",
        help="Fail if optional backends (onnxruntime) are missing",
    )
    args = parser.parse_args()

    maybe_reexec_in_project_venv()
    configure_runtime()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt = load_pi05_policy(args.checkpoint_dir)
    policy._model = policy._model.to(device)
    policy._model.eval()
    wrapper = PI05SampleActionsWrapper(policy._model, device=device, num_steps=args.num_steps).to(device).eval()
    inputs = prepare_inputs_from_policy(policy, device=device, seed=args.seed)

    logger.info("Benchmark checkpoint: %s", ckpt)
    torch_out, torch_t = benchmark_torch(wrapper, inputs, args.warmup, args.iters)
    logger.info("PyTorch median: %.3f ms", np.median(torch_t))
    torch_ref_cpu = torch_out.detach().cpu()

    ort_out = None
    ort_t = None
    ort_provider = None
    ort_metrics = None
    try:
        ort_out, ort_t, ort_provider = benchmark_onnxruntime(args.onnx, inputs, args.warmup, args.iters)
        ort_metrics = tensor_metrics(torch_ref_cpu, ort_out.cpu())
        logger.info("ONNX Runtime provider: %s", ort_provider)
        logger.info("ONNX Runtime median: %.3f ms", np.median(ort_t))
    except ModuleNotFoundError:
        if args.strict_backends:
            raise RuntimeError("onnxruntime is not installed") from None
        logger.warning("onnxruntime is not installed; skipping ONNX Runtime benchmark")

    # Release PyTorch policy/model memory before TensorRT benchmark.
    del wrapper
    del policy
    del torch_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    trt_t = None
    trt_metrics = None
    if args.trt_engine:
        trt_out, trt_t = benchmark_trt(args.trt_engine, inputs, args.warmup, args.iters)
        trt_metrics = tensor_metrics(torch_ref_cpu, trt_out.cpu())
        logger.info("TensorRT median: %.3f ms", np.median(trt_t))

    print("\n" + "=" * 110)
    print("| Backend      | Median Latency (ms) | Throughput (Hz) | MAE vs PT | MaxAbs vs PT | RelL2 vs PT | Cosine |")
    print("|--------------|---------------------|-----------------|-----------|--------------|-------------|--------|")
    print(
        f"| PyTorch      | {np.median(torch_t):.3f} | {1000.0/np.median(torch_t):.2f} | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.000000 |"
    )
    if ort_t is not None and ort_metrics is not None:
        print(
            f"| ONNXRuntime  | {np.median(ort_t):.3f} | {1000.0/np.median(ort_t):.2f} | {ort_metrics['mae']:.3e} | {ort_metrics['max_abs']:.3e} | {ort_metrics['rel_l2']:.3e} | {ort_metrics['cosine']:.6f} |"
        )
    else:
        print("| ONNXRuntime  | N/A | N/A | N/A | N/A | N/A | N/A |")
    if trt_t is not None and trt_metrics is not None:
        print(
            f"| TensorRT     | {np.median(trt_t):.3f} | {1000.0/np.median(trt_t):.2f} | {trt_metrics['mae']:.3e} | {trt_metrics['max_abs']:.3e} | {trt_metrics['rel_l2']:.3e} | {trt_metrics['cosine']:.6f} |"
        )
    elif args.trt_engine:
        raise RuntimeError("TensorRT benchmark requested but no TRT result produced")
    print("=" * 110)


if __name__ == "__main__":
    main()
