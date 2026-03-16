cmake_minimum_required(VERSION 3.18)
project(pi05_trt_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(_TRT_HINTS
  ${TensorRT_ROOT}
  $ENV{TensorRT_ROOT}
  /home/ace/Downloads/deploy_tmp/TensorRT-10.10.0.31
)

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS
    ${_TRT_HINTS}
    /usr/include/x86_64-linux-gnu
    /usr/include
    /usr/local/include
  PATH_SUFFIXES
    include
)

find_library(TENSORRT_LIBRARY
  NAMES nvinfer
  HINTS
    ${_TRT_HINTS}
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/lib
  PATH_SUFFIXES
    lib
    targets/x86_64-linux-gnu/lib
)

find_library(TENSORRT_PLUGIN_LIBRARY
  NAMES nvinfer_plugin
  HINTS
    ${_TRT_HINTS}
    /usr/lib/x86_64-linux-gnu
    /usr/lib
    /usr/local/lib
  PATH_SUFFIXES
    lib
    targets/x86_64-linux-gnu/lib
)

if(NOT TENSORRT_INCLUDE_DIR)
  message(FATAL_ERROR "TensorRT headers not found. Set TensorRT_ROOT or install TensorRT dev package.")
endif()

if(NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT library (nvinfer) not found. Set TensorRT_ROOT or install TensorRT runtime/dev package.")
endif()

add_executable(pi05_trt_runner src/main.cpp)

target_include_directories(pi05_trt_runner PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(pi05_trt_runner PRIVATE ${TENSORRT_LIBRARY} CUDA::cudart)

if(TENSORRT_PLUGIN_LIBRARY)
  target_link_libraries(pi05_trt_runner PRIVATE ${TENSORRT_PLUGIN_LIBRARY})
endif()

get_filename_component(_trt_lib_dir "${TENSORRT_LIBRARY}" DIRECTORY)
set_target_properties(pi05_trt_runner PROPERTIES
  BUILD_RPATH "${_trt_lib_dir}"
  INSTALL_RPATH "${_trt_lib_dir}"
)

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib    : ${TENSORRT_LIBRARY}")
if(TENSORRT_PLUGIN_LIBRARY)
  message(STATUS "TensorRT plugin : ${TENSORRT_PLUGIN_LIBRARY}")
endif()
