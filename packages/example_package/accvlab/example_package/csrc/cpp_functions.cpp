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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <numeric>
#include <algorithm>
#include <iostream>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

double vector_sum_cpu(torch::Tensor vec) {
    CHECK_INPUT(vec);
    TORCH_CHECK(vec.dtype() == torch::kFloat64 || vec.dtype() == torch::kFloat32,
                "Input tensor must be float32 or float64");
    TORCH_CHECK(vec.dim() == 1, "Input must be a 1D tensor");

    // Convert to CPU if needed and sum
    auto cpu_vec = vec.cpu();
    if (vec.dtype() == torch::kFloat64) {
        return cpu_vec.sum().item<double>();
    } else {
        return static_cast<double>(cpu_vec.sum().item<float>());
    }
}

torch::Tensor matrix_transpose_cpu(torch::Tensor matrix) {
    CHECK_INPUT(matrix);
    TORCH_CHECK(matrix.dim() == 2, "Input must be a 2D tensor (matrix)");

    // Use PyTorch's built-in transpose operation
    return matrix.transpose(0, 1);
}

void print_build_info_cpu() {
    std::cout << "ACCV-Lab Examples - C++ Extension" << std::endl;
    std::cout << "Compiled with: " << __VERSION__ << std::endl;
    std::cout << "C++ Standard: " << __cplusplus << std::endl;

#ifdef DEBUG
    std::cout << "Build Type: Debug" << std::endl;
#else
    std::cout << "Build Type: Release" << std::endl;
#endif
}

// Additional CPU tensor operations
torch::Tensor vector_multiply_cpu(torch::Tensor a, torch::Tensor b) {
    CHECK_INPUT(a);
    CHECK_INPUT(b);

    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    TORCH_CHECK(a.dtype() == b.dtype(), "Input tensors must have the same dtype");

    // Element-wise multiplication using PyTorch operations
    return a * b;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ACCV-Lab Examples C++ Extension";
    m.def("vector_sum", &vector_sum_cpu, "Sum vector elements using C++");
    m.def("matrix_transpose", &matrix_transpose_cpu, "Transpose a matrix using C++");
    m.def("print_build_info", &print_build_info_cpu, "Print build information for C++ extension");
    m.def("vector_multiply", &vector_multiply_cpu, "Element-wise vector multiplication using C++");
}