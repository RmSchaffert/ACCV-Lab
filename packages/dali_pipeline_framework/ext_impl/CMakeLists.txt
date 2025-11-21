# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.10)
include(CMakePrintHelpers)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    # Fallback list for older CMake versions without 'native'
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
  endif()
endif()

project(get_affine_transform_plugin LANGUAGES CUDA CXX C)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

cmake_print_variables(lapack_libraries)
include_directories(SYSTEM "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

execute_process(
        COMMAND python3 -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
        OUTPUT_VARIABLE DALI_LIB_DIR)
string(STRIP "${DALI_LIB_DIR}" DALI_LIB_DIR)

execute_process(
        COMMAND python3 -c "import nvidia.dali as dali; print(dali.sysconfig.get_include_dir())"
        OUTPUT_VARIABLE DALI_INCLUDE_DIR)
string(STRIP "${DALI_INCLUDE_DIR}" DALI_INCLUDE_DIR)

execute_process(
        COMMAND python3 -c "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
string(STRIP "${DALI_COMPILE_FLAGS}" DALI_COMPILE_FLAGS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS} ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${DALI_COMPILE_FLAGS} ")
include_directories("${DALI_INCLUDE_DIR}")
link_directories("${DALI_LIB_DIR}")

add_library(_draw_gaussians SHARED DrawGaussians.cc)
target_link_libraries(_draw_gaussians dali)

install(TARGETS _draw_gaussians
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)






