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

#include <torch/extension.h>

// Declare C++ functions
double vector_sum_cpu(torch::Tensor vec);
torch::Tensor matrix_transpose_cpu(torch::Tensor matrix);
void print_build_info_cpu();
torch::Tensor vector_multiply_cpu(torch::Tensor a, torch::Tensor b);

// Declare CUDA functions
torch::Tensor vector_multiply_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor reduce_sum_cuda(torch::Tensor input);