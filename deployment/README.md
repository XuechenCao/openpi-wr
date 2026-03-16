# OpenPI pi05 Deployment (ONNX + TensorRT)

This folder contains scripts to:
- export `pi05` PyTorch policy (`model.safetensors`) to ONNX
- build a TensorRT engine
- benchmark speed and accuracy vs PyTorch

## Files
- `export_onnx_pi05_pytorch.py`: export ONNX and save sample I/O
- `build_tensorrt_engine_pi05.py`: build TensorRT engine from ONNX
- `benchmark_pi05_onnx_trt.py`: benchmark PyTorch / ONNX Runtime / TensorRT
- `pi05_deploy_utils.py`: shared loading + input preparation helpers

## Requirements
- Python env with OpenPI deps (`torch`, `jax`, etc.)
- TensorRT runtime + Python bindings (`import tensorrt`)
- Optional: ONNX Runtime (`import onnxruntime`) for ORT row in benchmark

## 1) Export ONNX
```bash
cd /home/ace/Downloads/deploy_tmp/openpi

python3 examples/deployment/export_onnx_pi05_pytorch.py \
  --checkpoint_dir /home/ace/Downloads/deploy_tmp/openpi/examples/deployment \
  --output_dir /home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx \
  --onnx_name pi05_sample_actions.onnx \
  --num_steps 10
```

Artifacts:
- `onnx/pi05_sample_actions.onnx`
- `onnx/sample_io.npz` (inputs + PyTorch reference output)

## 2) Build TensorRT Engine (Recommended: BF16 for accuracy)
```bash
python3 examples/deployment/build_tensorrt_engine_pi05.py \
  --onnx /home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions.onnx \
  --engine /home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions_bf16_v2.trt \
  --precision bf16
```

Note:
- Default prompt profile is fixed to `200` tokens to match exported graph constraints.
- BF16 gives much better output fidelity than FP16 on this model.

## 3) Benchmark (Speed + Accuracy)
```bash
python3 examples/deployment/benchmark_pi05_onnx_trt.py \
  --checkpoint_dir /home/ace/Downloads/deploy_tmp/openpi/examples/deployment \
  --onnx /home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions.onnx \
  --trt_engine /home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions_bf16_v2.trt \
  --warmup 3 \
  --iters 20
```

Output table columns:
- `Median Latency (ms)`: median per-inference latency
- `Throughput (Hz)`: `1000 / median_latency`
- `MAE / MaxAbs / RelL2 / Cosine`: error metrics vs PyTorch output

## Troubleshooting
- If benchmark fails with TensorRT import error:
  - run it in the env where `python3 -c "import tensorrt"` succeeds
- If ONNX Runtime row is `N/A`:
  - install `onnxruntime` (or `onnxruntime-gpu`)
- TensorRT build can take several minutes for this model.

## Verified Result (Torch vs TRT)
Recent run (20 iters, warmup 3):
- PyTorch median latency: `105.353 ms` (`9.49 Hz`)
- TensorRT median latency: `66.056 ms` (`15.14 Hz`)
- TensorRT speedup vs PyTorch: `~1.59x`
- Accuracy vs PyTorch: `MAE=7.740e-03`, `RelL2=7.208e-02`, `Cosine=0.997407`
