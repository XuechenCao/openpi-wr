# PI05 TensorRT C++ Mini Project

This project runs PI05 TensorRT engine inference in C++.

Important version note:
- `pi05_sample_actions_bf16_v2.trt` was built with a newer TensorRT and cannot be deserialized by TensorRT 10.10.
- Rebuild a compatible engine first (steps below), then run C++.

## Files

- `CMakeLists.txt`
- `src/main.cpp`
- `build_engine_trt1010.sh`
- `run_pi05_trt_cpp.sh`

## End-to-End (TensorRT 10.10)

```bash
cd /home/ace/Downloads/deploy_tmp/openpi/examples/deployment/deploy_cpp

# 1) Build a TensorRT-10.10-compatible engine from ONNX
./build_engine_trt1010.sh

# 2) Build and run C++ benchmark
./run_pi05_trt_cpp.sh
```

Defaults:
- `TRT_ROOT=/home/ace/Downloads/deploy_tmp/TensorRT-10.10.0.31`
- Engine output:
  `/home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions_bf16_v2_trt1010.trt`
- Runner args: `warmup=1`, `iters=5`

## Manual Commands

```bash
cd /home/ace/Downloads/deploy_tmp/openpi/examples/deployment/deploy_cpp

cmake -S . -B build -DTensorRT_ROOT=/home/ace/Downloads/deploy_tmp/TensorRT-10.10.0.31
cmake --build build -j

export LD_LIBRARY_PATH=/home/ace/Downloads/deploy_tmp/TensorRT-10.10.0.31/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1

./build/pi05_trt_runner \
  /home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions_bf16_v2_trt1010.trt \
  1 5
```

## Verified Result

Validated on this machine with TensorRT `10.10.0`:
- Engine built successfully (`~6223 MiB`)
- C++ run successful (`11` I/O tensors detected)
- Median latency: `61.544 ms`
- Throughput: `16.249 Hz`
