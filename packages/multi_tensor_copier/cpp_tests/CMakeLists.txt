# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED ACCVLAB_PACKAGE_CMAKE_VERSION)
    set(ACCVLAB_PACKAGE_CMAKE_VERSION "0.0.0")
endif()
project(
    accvlab_multi_tensor_copier_cpp_tests
    VERSION ${ACCVLAB_PACKAGE_CMAKE_VERSION}
    LANGUAGES CXX CUDA
)

if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
    COMMAND "python3" -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake'))"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

find_package(CUDA REQUIRED)

if(DEFINED ACCVLAB_TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "${ACCVLAB_TORCH_CUDA_ARCH_LIST}")
endif()
find_package(Torch REQUIRED)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib REQUIRED)

option(
    ACCVLAB_MULTI_TENSOR_COPIER_BUILD_CPP_TESTS
    "Build multi_tensor_copier native C++ tests"
    OFF
)

if(ACCVLAB_MULTI_TENSOR_COPIER_BUILD_CPP_TESTS)
    enable_testing()
    add_subdirectory(utest)
endif()
