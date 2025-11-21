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

#include "examples.h"
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

// CUDA kernel for vector multiplication
__global__ void vector_multiply_kernel(float* a, float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

// CUDA kernel for reduction sum
__global__ void reduce_sum_kernel(float* input, float* result, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

// C++ wrapper for vector multiplication
torch::Tensor vector_multiply_cuda(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on CUDA device");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on CUDA device");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "Input tensor a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "Input tensor b must be float32");

    auto result = torch::empty_like(a);
    int size = a.numel();
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;

    vector_multiply_kernel<<<num_blocks, threads_per_block>>>(a.data_ptr<float>(), b.data_ptr<float>(),
                                                              result.data_ptr<float>(), size);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return result;
}

// C++ wrapper for reduction sum
torch::Tensor reduce_sum_cuda(torch::Tensor input) {
    CHECK_INPUT(input);
    TORCH_CHECK(input.device().is_cuda(), "Input tensor must be on CUDA device");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");

    int size = input.numel();
    int threads_per_block = 256;
    int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    auto result = torch::zeros({num_blocks}, torch::kFloat32).cuda();

    reduce_sum_kernel<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(
        input.data_ptr<float>(), result.data_ptr<float>(), size);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return result.sum();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ACCV-Lab Examples CUDA Extension";
    m.def("vector_multiply", &vector_multiply_cuda, "Element-wise vector multiplication using CUDA");
    m.def("reduce_sum", &reduce_sum_cuda, "Parallel reduction sum using CUDA");
}