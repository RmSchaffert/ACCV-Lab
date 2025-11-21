/*
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "FixedSizeVideoReaderMap.hpp"
#include "GPUMemoryPool.hpp"
#include "PyNvVideoReader.hpp"
#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef IS_DEBUG_BUILD
class __attribute__((visibility("default"))) PyNvSampleReader {
#else
class PyNvSampleReader {
#endif
   public:
    // Currently, we assume that for each file, the frame_ids to be extracted have
    // no duplication.
    PyNvSampleReader(int num_of_set, int num_of_file, int iGpu, bool bSuppressNoColorRangeWarning = false);

    ~PyNvSampleReader();

    void clearAllReaders();

    std::vector<RGBFrame> run_rgb_out(const std::vector<std::string>& filepaths,
                                      const std::vector<int> frame_ids, bool as_bgr);
    std::vector<DecodedFrameExt> run(const std::vector<std::string>& filepaths,
                                     const std::vector<int> frame_ids);

   private:
    bool suppress_no_color_range_given_warning = false;
    bool destroy_context = false;
    CUcontext cu_context = NULL;
    CUstream cu_stream = NULL;
    int gpu_id = 0;
    int num_of_file = 0;
    int num_of_set = 0;

    std::vector<FixedSizeVideoReaderMap> VideoReaderMap;
    GPUMemoryPool gpu_mem_pool;
};
