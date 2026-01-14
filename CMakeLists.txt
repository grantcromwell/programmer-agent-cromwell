cmake_minimum_required(VERSION 3.18)
project(cromwell_agent VERSION 1.0.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mtune=native -O3 -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2 /O2")
endif()

# Find dependencies
find_package(Threads REQUIRED)

# Core library
add_library(cromwell_core STATIC
    src/core/transformer.cpp
    src/core/attention.cpp
    src/core/mlp.cpp
    src/core/embeddings.cpp
    src/core/config.cpp
)

target_include_directories(cromwell_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cromwell_core PUBLIC
    Threads::Threads
)

# Operations library
add_library(cromwell_ops STATIC
    src/ops/gemm.cpp
    src/ops/attention_ops.cpp
    src/ops/layer_norm.cpp
    src/ops/activation.cpp
)

target_include_directories(cromwell_ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cromwell_ops PUBLIC
    cromwell_core
)

# Inference library
add_library(cromwell_inference STATIC
    src/inference/sampler.cpp
    src/inference/kv_cache.cpp
    src/inference/generator.cpp
)

target_include_directories(cromwell_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cromwell_inference PUBLIC
    cromwell_core
    cromwell_ops
)

# I/O library
add_library(cromwell_io STATIC
    src/io/tokenizer.cpp
    src/io/file_io.cpp
    src/io/diff.cpp
    src/io/context_manager.cpp
)

target_include_directories(cromwell_io PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cromwell_io PUBLIC
    cromwell_core
)

# Python bindings
find_package(pybind11 REQUIRED)

pybind11_add_module(cromwell_python
    src/python/bindings.cpp
)

target_include_directories(cromwell_python PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(cromwell_python PRIVATE
    cromwell_core
    cromwell_ops
    cromwell_inference
    cromwell_io
)

# Tests
enable_testing()

add_subdirectory(tests)

# Installation
install(TARGETS cromwell_core cromwell_ops cromwell_inference cromwell_io
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY src/
    DESTINATION include/cromwell
    FILES_MATCHING PATTERN "*.h"
)

# Examples
add_executable(example_attention
    examples/attention.cpp
)

target_link_libraries(example_attention
    cromwell_core
    cromwell_ops
)

add_executable(example_generation
    examples/generation.cpp
)

target_link_libraries(example_generation
    cromwell_inference
)

# Benchmarks
add_executable(benchmark_attention
    benchmarks/benchmark_attention.cpp
)

target_link_libraries(benchmark_attention
    cromwell_core
    cromwell_ops
)

add_executable(benchmark_gemm
    benchmarks/benchmark_gemm.cpp
)

target_link_libraries(benchmark_gemm
    cromwell_ops
)
