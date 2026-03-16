#!/usr/bin/env bash
set -euo pipefail

TRT_ROOT="${TRT_ROOT:-/home/ace/Downloads/deploy_tmp/TensorRT-10.10.0.31}"
ONNX_PATH="${ONNX_PATH:-/home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions.onnx}"
ENGINE_PATH="${ENGINE_PATH:-/home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions_bf16_v2_trt1010.trt}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

if [[ ! -f "${TRT_ROOT}/bin/trtexec" ]]; then
  echo "trtexec not found at ${TRT_ROOT}/bin/trtexec"
  exit 1
fi

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ONNX file not found: ${ONNX_PATH}"
  exit 1
fi

export LD_LIBRARY_PATH="${TRT_ROOT}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES

SHAPES="state:1x32,base_image:1x3x224x224,left_wrist_image:1x3x224x224,right_wrist_image:1x3x224x224,base_mask:1,left_mask:1,right_mask:1,tokenized_prompt:1x200,tokenized_prompt_mask:1x200,noise:1x10x32"

"${TRT_ROOT}/bin/trtexec" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${ENGINE_PATH}" \
  --minShapes="${SHAPES}" \
  --optShapes="${SHAPES}" \
  --maxShapes="${SHAPES}" \
  --bf16 \
  --skipInference

echo "Engine created: ${ENGINE_PATH}"
