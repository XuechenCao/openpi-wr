#!/usr/bin/env python3
import argparse
import logging
import os
import time

import numpy as np
import tensorrt as trt


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_FLOAT_DTYPES = {trt.DataType.FLOAT, trt.DataType.HALF, trt.DataType.BF16}


def _torch_dtype_for_trt(dtype: trt.DataType):
    import torch

    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.BOOL: torch.bool,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported TRT input dtype for calibration: {dtype}")
    return mapping[dtype]


class SampleIoEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator backed by examples/deployment ONNX sample_io.npz."""

    def __init__(self, sample_io_path: str, network_inputs: list, cache_path: str):
        super().__init__()
        import torch

        if not os.path.isfile(sample_io_path):
            raise FileNotFoundError(f"INT8 calibration sample file not found: {sample_io_path}")

        self._cache_path = cache_path
        self._sent = False
        self._device_tensors = {}
        npz = np.load(sample_io_path)
        for inp in network_inputs:
            name = inp.name
            if name not in npz.files:
                raise KeyError(f"Missing input '{name}' in calibration sample: {sample_io_path}")
            arr = npz[name]
            torch_dtype = _torch_dtype_for_trt(inp.dtype)
            ten = torch.from_numpy(arr).to(dtype=torch_dtype, device="cuda").contiguous()
            self._device_tensors[name] = ten
            logger.info("Calibration input %-22s shape=%s dtype=%s", name, tuple(arr.shape), arr.dtype)

    def get_batch_size(self) -> int:
        # Sample IO is exported with batch=1.
        return 1

    def get_batch(self, names) -> list[int] | None:
        if self._sent:
            return None
        self._sent = True
        return [int(self._device_tensors[n].data_ptr()) for n in names]

    def read_calibration_cache(self):
        if os.path.isfile(self._cache_path):
            logger.info("Using existing INT8 calibration cache: %s", self._cache_path)
            with open(self._cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        os.makedirs(os.path.dirname(self._cache_path) or ".", exist_ok=True)
        with open(self._cache_path, "wb") as f:
            f.write(cache)
        logger.info("Wrote INT8 calibration cache: %s", self._cache_path)


def _is_float_output_layer(layer: trt.ILayer) -> bool:
    if layer.num_outputs == 0:
        return False
    for i in range(layer.num_outputs):
        out = layer.get_output(i)
        if out is None or out.dtype not in _FLOAT_DTYPES:
            return False
    return True


def _layer_has_gemma_expert_weight_input(layer: trt.ILayer) -> bool:
    for i in range(layer.num_inputs):
        inp = layer.get_input(i)
        if inp is not None and "gemma_expert" in (inp.name or ""):
            return True
    return False


def _apply_gemma_int8_other_bf16(
    network: trt.INetworkDefinition,
    config: trt.IBuilderConfig,
    profile: trt.IOptimizationProfile,
    sample_io_path: str,
    calib_cache_path: str,
):
    config.set_flag(trt.BuilderFlag.BF16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)

    gemma_int8_layers = 0
    other_bf16_layers = 0
    skipped_non_float = 0

    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if not _is_float_output_layer(layer):
            skipped_non_float += 1
            continue
        if _layer_has_gemma_expert_weight_input(layer):
            layer.precision = trt.DataType.INT8
            gemma_int8_layers += 1
        else:
            layer.precision = trt.DataType.BF16
            other_bf16_layers += 1

    logger.info(
        "Applied mixed precision constraints: gemma_int8_layers=%d other_bf16_layers=%d skipped_non_float=%d",
        gemma_int8_layers,
        other_bf16_layers,
        skipped_non_float,
    )

    calibrator = SampleIoEntropyCalibrator(
        sample_io_path=sample_io_path,
        network_inputs=[network.get_input(i) for i in range(network.num_inputs)],
        cache_path=calib_cache_path,
    )
    config.int8_calibrator = calibrator
    config.set_calibration_profile(profile)
    return calibrator


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str,
    workspace_mb: int,
    batch_min: int,
    batch_opt: int,
    batch_max: int,
    prompt_min: int,
    prompt_opt: int,
    prompt_max: int,
    int8_sample_io: str | None,
    int8_calib_cache: str | None,
):
    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)

    if not parser.parse_from_file(onnx_path):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("Failed to parse ONNX:\n" + "\n".join(errs))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1024**2))
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision == "fp8":
        config.set_flag(trt.BuilderFlag.FP8)

    profile = builder.create_optimization_profile()

    def _shape_triplet_from_network(name: str, shape: tuple[int, ...]):
        # Use ONNX-declared dimensions, replacing only dynamic axes.
        min_shape, opt_shape, max_shape = [], [], []
        for axis, d in enumerate(shape):
            if d != -1:
                min_shape.append(d)
                opt_shape.append(d)
                max_shape.append(d)
                continue

            # Dynamic axes: batch and prompt length.
            if axis == 0:
                min_shape.append(batch_min)
                opt_shape.append(batch_opt)
                max_shape.append(batch_max)
            elif name in {"tokenized_prompt", "tokenized_prompt_mask"} and axis == 1:
                min_shape.append(prompt_min)
                opt_shape.append(prompt_opt)
                max_shape.append(prompt_max)
            else:
                # Any other dynamic axis defaults to batch profile.
                min_shape.append(batch_min)
                opt_shape.append(batch_opt)
                max_shape.append(batch_max)
        return tuple(min_shape), tuple(opt_shape), tuple(max_shape)
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        name = inp.name
        mn, op, mx = _shape_triplet_from_network(name, tuple(inp.shape))
        profile.set_shape(name, mn, op, mx)
        logger.info("Profile %-22s min=%s opt=%s max=%s", name, mn, op, mx)
    config.add_optimization_profile(profile)

    calibrator = None
    if precision == "mixed_gemma_int8_bf16":
        sample_io_path = int8_sample_io or os.path.join(os.path.dirname(os.path.abspath(onnx_path)), "sample_io.npz")
        calib_cache_path = int8_calib_cache or os.path.join(
            os.path.dirname(os.path.abspath(engine_path)), "gemma_int8.calib"
        )
        calibrator = _apply_gemma_int8_other_bf16(
            network=network,
            config=config,
            profile=profile,
            sample_io_path=sample_io_path,
            calib_cache_path=calib_cache_path,
        )

    logger.info("Building TensorRT engine...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    dt = time.time() - t0

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized)
    logger.info("Engine saved: %s (%.2f MB, %.1fs)", engine_path, os.path.getsize(engine_path) / (1024**2), dt)
    if calibrator is not None:
        # Keep calibrator alive until build + serialization completes.
        _ = calibrator


def main():
    parser = argparse.ArgumentParser(description="Build TensorRT engine from OpenPI pi05 ONNX")
    parser.add_argument("--onnx", type=str, required=True)
    parser.add_argument("--engine", type=str, required=True)
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16", "fp8", "mixed_gemma_int8_bf16"],
        help="Use mixed_gemma_int8_bf16 to quantize Gemma expert-weighted layers to INT8 and keep other float layers BF16.",
    )
    parser.add_argument(
        "--int8-sample-io",
        type=str,
        default=None,
        help="Path to sample_io.npz used for INT8 calibration (required for mixed_gemma_int8_bf16 mode).",
    )
    parser.add_argument(
        "--int8-calib-cache",
        type=str,
        default=None,
        help="Path to calibration cache file for mixed_gemma_int8_bf16 mode.",
    )
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--batch-min", type=int, default=1)
    parser.add_argument("--batch-opt", type=int, default=1)
    parser.add_argument("--batch-max", type=int, default=1)
    parser.add_argument("--prompt-len-min", type=int, default=200)
    parser.add_argument("--prompt-len-opt", type=int, default=200)
    parser.add_argument("--prompt-len-max", type=int, default=200)
    args = parser.parse_args()

    if not (args.batch_min <= args.batch_opt <= args.batch_max):
        raise ValueError("Require batch-min <= batch-opt <= batch-max")
    if not (args.prompt_len_min <= args.prompt_len_opt <= args.prompt_len_max):
        raise ValueError("Require prompt-len-min <= prompt-len-opt <= prompt-len-max")

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        workspace_mb=args.workspace,
        batch_min=args.batch_min,
        batch_opt=args.batch_opt,
        batch_max=args.batch_max,
        prompt_min=args.prompt_len_min,
        prompt_opt=args.prompt_len_opt,
        prompt_max=args.prompt_len_max,
        int8_sample_io=args.int8_sample_io,
        int8_calib_cache=args.int8_calib_cache,
    )


if __name__ == "__main__":
    main()
