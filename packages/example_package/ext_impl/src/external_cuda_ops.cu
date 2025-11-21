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
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/external_cuda_ops.h"

// CUDA kernel for vector addition
__global__ void vector_add_kernel(float* a, float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

// CUDA kernel for vector scaling
__global__ void vector_scale_kernel(float* input, float* result, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = input[idx] * scale;
    }
}

// C++ wrapper for vector addition
torch::Tensor external_vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    // Check inputs
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on CUDA device");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on CUDA device");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");

    // Create output tensor
    auto result = torch::empty_like(a);

    // Get size and set up CUDA launch parameters
    int size = a.numel();
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    vector_add_kernel<<<num_blocks, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                                         result.data_ptr<float>(), size);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return result;
}

// C++ wrapper for vector scaling
torch::Tensor external_vector_scale_cuda(torch::Tensor input, float scale) {
    // Check input
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");

    // Create output tensor
    auto result = torch::empty_like(input);

    // Get size and set up CUDA launch parameters
    int size = input.numel();
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    vector_scale_kernel<<<num_blocks, threads_per_block>>>(input.data_ptr<float>(), result.data_ptr<float>(),
                                                           scale, size);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return result;
}