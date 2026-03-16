#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRT_ROOT="${TRT_ROOT:-/home/ace/Downloads/deploy_tmp/TensorRT-10.10.0.31}"
ENGINE_PATH="${1:-/home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions_bf16_v2_trt1010.trt}"
WARMUP="${2:-1}"
ITERS="${3:-5}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" -DTensorRT_ROOT="${TRT_ROOT}"
cmake --build "${SCRIPT_DIR}/build" -j

export LD_LIBRARY_PATH="${TRT_ROOT}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES

"${SCRIPT_DIR}/build/pi05_trt_runner" "${ENGINE_PATH}" "${WARMUP}" "${ITERS}"
