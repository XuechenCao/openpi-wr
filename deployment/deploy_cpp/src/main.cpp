#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << msg << '\n';
    }
  }
};

template <typename T>
struct TrtDeleter {
  void operator()(T* obj) const {
    if (!obj) {
      return;
    }
#if NV_TENSORRT_MAJOR >= 8
    delete obj;
#else
    obj->destroy();
#endif
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

void checkCuda(cudaError_t status, const std::string& what) {
  if (status != cudaSuccess) {
    std::cerr << "CUDA error at " << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(1);
  }
}

std::vector<char> readFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    std::cerr << "Failed to open engine: " << path << '\n';
    std::exit(1);
  }
  in.seekg(0, std::ios::end);
  const auto size = static_cast<size_t>(in.tellg());
  in.seekg(0, std::ios::beg);
  std::vector<char> data(size);
  in.read(data.data(), static_cast<std::streamsize>(size));
  return data;
}

int64_t volume(const nvinfer1::Dims& dims) {
  if (dims.nbDims < 0) {
    return 0;
  }
  int64_t v = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      return 0;
    }
    v *= dims.d[i];
  }
  return v;
}

size_t elementSize(nvinfer1::DataType dtype) {
  using nvinfer1::DataType;
  switch (dtype) {
    case DataType::kFLOAT:
      return 4;
    case DataType::kHALF:
      return 2;
    case DataType::kINT8:
      return 1;
    case DataType::kINT32:
      return 4;
    case DataType::kBOOL:
      return 1;
    case DataType::kUINT8:
      return 1;
    case DataType::kINT64:
      return 8;
    case DataType::kBF16:
      return 2;
    default:
      return 0;
  }
}

std::string dimsToString(const nvinfer1::Dims& dims) {
  std::string out = "(";
  for (int i = 0; i < dims.nbDims; ++i) {
    out += std::to_string(dims.d[i]);
    if (i + 1 < dims.nbDims) {
      out += ", ";
    }
  }
  out += ")";
  return out;
}

struct TensorBuffer {
  std::string name;
  nvinfer1::TensorIOMode mode{};
  nvinfer1::DataType dtype{};
  nvinfer1::Dims shape{};
  size_t bytes{0};
  std::vector<uint8_t> host;
  void* device{nullptr};
};

void fillInput(TensorBuffer& t) {
  const int64_t n = volume(t.shape);
  if (n <= 0) {
    std::cerr << "Invalid input shape for " << t.name << ": " << dimsToString(t.shape) << '\n';
    std::exit(1);
  }

  if (t.dtype == nvinfer1::DataType::kFLOAT) {
    auto* ptr = reinterpret_cast<float*>(t.host.data());
    for (int64_t i = 0; i < n; ++i) {
      ptr[i] = 0.0f;
    }
    if (t.name == "noise") {
      for (int64_t i = 0; i < n; ++i) {
        ptr[i] = static_cast<float>((i % 17) * 0.01f);
      }
    }
    if (t.name.find("image") != std::string::npos) {
      for (int64_t i = 0; i < n; ++i) {
        ptr[i] = 0.5f;
      }
    }
    return;
  }

  if (t.dtype == nvinfer1::DataType::kBOOL) {
    auto* ptr = reinterpret_cast<bool*>(t.host.data());
    for (int64_t i = 0; i < n; ++i) {
      ptr[i] = true;
    }
    if (t.name == "tokenized_prompt_mask") {
      for (int64_t i = 0; i < n; ++i) {
        ptr[i] = (i < 16);
      }
    }
    return;
  }

  if (t.dtype == nvinfer1::DataType::kINT64) {
    auto* ptr = reinterpret_cast<int64_t*>(t.host.data());
    for (int64_t i = 0; i < n; ++i) {
      ptr[i] = (i % 128) + 1;
    }
    return;
  }

  std::cerr << "Unsupported input dtype for " << t.name << '\n';
  std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
  std::string enginePath =
      "/home/ace/Downloads/deploy_tmp/openpi/examples/deployment/onnx/pi05_sample_actions_bf16_v2_trt1010.trt";
  int warmup = 3;
  int iters = 20;

  if (argc > 1) {
    enginePath = argv[1];
  }
  if (argc > 2) {
    warmup = std::atoi(argv[2]);
  }
  if (argc > 3) {
    iters = std::atoi(argv[3]);
  }

  TrtLogger logger;
  initLibNvInferPlugins(&logger, "");

  const auto engineBytes = readFile(enginePath);
  TrtUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
  if (!runtime) {
    std::cerr << "Failed to create TensorRT runtime\n";
    return 1;
  }

  TrtUniquePtr<nvinfer1::ICudaEngine> engine{
      runtime->deserializeCudaEngine(engineBytes.data(), engineBytes.size())};
  if (!engine) {
    std::cerr << "Failed to deserialize engine: " << enginePath << '\n';
    return 1;
  }

  TrtUniquePtr<nvinfer1::IExecutionContext> context{engine->createExecutionContext()};
  if (!context) {
    std::cerr << "Failed to create execution context\n";
    return 1;
  }

  std::vector<TensorBuffer> tensors;
  tensors.reserve(engine->getNbIOTensors());

  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    const auto mode = engine->getTensorIOMode(name);
    const auto shape = engine->getTensorShape(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      if (!context->setInputShape(name, shape)) {
        std::cerr << "Failed to set input shape for " << name << " " << dimsToString(shape) << '\n';
        return 1;
      }
    }
  }

  std::cout << "Loaded engine: " << enginePath << '\n';
  std::cout << "I/O tensors: " << engine->getNbIOTensors() << '\n';

  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    TensorBuffer t;
    t.name = name;
    t.mode = engine->getTensorIOMode(name);
    t.dtype = engine->getTensorDataType(name);
    t.shape = context->getTensorShape(name);

    const int64_t elems = volume(t.shape);
    const size_t elemSize = elementSize(t.dtype);
    if (elems <= 0 || elemSize == 0) {
      std::cerr << "Invalid tensor info for " << t.name << ", shape=" << dimsToString(t.shape) << '\n';
      return 1;
    }
    t.bytes = static_cast<size_t>(elems) * elemSize;
    t.host.resize(t.bytes);

    checkCuda(cudaMalloc(&t.device, t.bytes), "cudaMalloc(" + t.name + ")");
    if (!context->setTensorAddress(t.name.c_str(), t.device)) {
      std::cerr << "Failed to set tensor address for " << t.name << '\n';
      return 1;
    }

    std::cout << " - " << t.name << " | mode=" << (t.mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT")
              << " | bytes=" << t.bytes << " | shape=" << dimsToString(t.shape) << '\n';

    if (t.mode == nvinfer1::TensorIOMode::kINPUT) {
      fillInput(t);
    }
    tensors.push_back(std::move(t));
  }

  cudaStream_t stream{};
  checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");

  auto copyInputs = [&]() {
    for (auto& t : tensors) {
      if (t.mode == nvinfer1::TensorIOMode::kINPUT) {
        checkCuda(cudaMemcpyAsync(t.device, t.host.data(), t.bytes, cudaMemcpyHostToDevice, stream),
                  "cudaMemcpyAsync H2D " + t.name);
      }
    }
  };

  auto copyOutputs = [&]() {
    for (auto& t : tensors) {
      if (t.mode == nvinfer1::TensorIOMode::kOUTPUT) {
        checkCuda(cudaMemcpyAsync(t.host.data(), t.device, t.bytes, cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync D2H " + t.name);
      }
    }
  };

  for (int i = 0; i < warmup; ++i) {
    copyInputs();
    if (!context->enqueueV3(stream)) {
      std::cerr << "enqueueV3 failed during warmup\n";
      return 1;
    }
    checkCuda(cudaStreamSynchronize(stream), "warmup sync");
  }

  std::vector<double> latMs;
  latMs.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    copyInputs();
    const auto t0 = std::chrono::high_resolution_clock::now();
    if (!context->enqueueV3(stream)) {
      std::cerr << "enqueueV3 failed at iter " << i << '\n';
      return 1;
    }
    checkCuda(cudaStreamSynchronize(stream), "iter sync");
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    latMs.push_back(ms);
  }

  copyOutputs();
  checkCuda(cudaStreamSynchronize(stream), "output sync");

  std::sort(latMs.begin(), latMs.end());
  const double median = latMs[latMs.size() / 2];
  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Median latency: " << median << " ms | Throughput: " << (1000.0 / median) << " Hz\n";

  for (const auto& t : tensors) {
    if (t.name == "actions") {
      const float* out = reinterpret_cast<const float*>(t.host.data());
      const int64_t n = volume(t.shape);
      const int64_t show = std::min<int64_t>(n, 10);
      std::cout << "actions[0:" << show << "]: ";
      for (int64_t i = 0; i < show; ++i) {
        std::cout << out[i] << (i + 1 < show ? ", " : "");
      }
      std::cout << '\n';
      break;
    }
  }

  for (auto& t : tensors) {
    cudaFree(t.device);
    t.device = nullptr;
  }
  cudaStreamDestroy(stream);
  return 0;
}
