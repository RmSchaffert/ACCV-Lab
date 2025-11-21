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

cmake_minimum_required(VERSION 3.21)
project(accvlab.on_demand_video_decoder)
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(DEFINED ENV{FFMPEG_DIR})
    set(FFMPEG_DIR "$ENV{FFMPEG_DIR}" CACHE PATH "path for ffmpeg installed")
else()
    set(FFMPEG_DIR "/usr/local" CACHE PATH "path for ffmpeg installed")
endif()

set(NV_FFMPEG_LIBRARIES  "${CMAKE_CURRENT_SOURCE_DIR}/src/VideoCodecSDKUtils/Interface" CACHE PATH "path for VideoCodecSDKUtils ffmpeg path")
set(NVTX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/NVTX/c/include" CACHE PATH "path for nvtx head file") 
set(DLPACK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dlpack" CACHE PATH "path for dlpack lib") 
set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11" CACHE PATH "path for pybind11 lib") 

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Set CMake variable
    set(IS_DEBUG_BUILD "true")
    # Set pre-processor macro
    add_definitions(-DIS_DEBUG_BUILD)
endif()

add_subdirectory(src)

#If we need to run with santizer, the way is ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=1 ./cpp_samples/run_decoding
#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

message("${CMAKE_BUILD_TYPE}")
