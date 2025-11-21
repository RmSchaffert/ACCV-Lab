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
#include <vector>

#include <stddef.h>
#include <stdint.h>

class GPUMemoryPool {
   public:
    GPUMemoryPool() = default;

    GPUMemoryPool(size_t size_hint);

    void* AddElement(size_t num_bytes);

    void EnsureSizeAndSoftReset(size_t num_bytes_to_store, bool shrink_if_smaller);

    void SoftRelease();

    void HardRelease();

    ~GPUMemoryPool();

   private:
    uint8_t* data_ = nullptr;
    size_t curr_size_ = 0;
    size_t allocated_size_ = 0;
};