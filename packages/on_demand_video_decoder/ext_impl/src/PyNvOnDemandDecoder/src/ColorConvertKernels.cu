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

#include "ColorConvertKernels.cuh"

#include <cuda.h>

#include "NvCodecUtils.h"

template <typename T>
__device__ __forceinline__ T max_func(T a, T b) {
    const T res = a > b ? a : b;
    return res;
}

template <typename T>
__device__ __forceinline__ T min_func(T a, T b) {
    const T res = a < b ? a : b;
    return res;
}

template <typename T>
__device__ __forceinline__ T cv_descale_func(T x, T n) {
    const T res = ((x + (1 << (n - 1))) >> n);
    return res;
}

template <typename T>
__device__ __forceinline__ T clamp_func(T val, T min, T max) {
    const T res = min_func(max_func(val, min), max);
    return res;
}

__device__ __forceinline__ void yuv42xxp_to_rgb_kernel(const int& Y, const int& U, const int& V, uint8_t& r,
                                                       uint8_t& g, uint8_t& b) {
    constexpr int ITUR_BT_601_CY = 1220542;
    constexpr int ITUR_BT_601_CUB = 2116026;
    constexpr int ITUR_BT_601_CUG = -409993;
    constexpr int ITUR_BT_601_CVG = -852492;
    constexpr int ITUR_BT_601_CVR = 1673527;
    constexpr int ITUR_BT_601_SHIFT = 20;

    // R = 1.164(Y - 16) + 1.596(V - 128)
    // G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
    // B = 1.164(Y - 16)                  + 2.018(U - 128)

    // R = (1220542(Y - 16) + 1673527(V - 128)                  + (1 << 19)) >> 20
    // G = (1220542(Y - 16) - 852492(V - 128) - 409993(U - 128) + (1 << 19)) >> 20
    // B = (1220542(Y - 16)                  + 2116026(U - 128) + (1 << 19)) >> 20
    const int C0 = ITUR_BT_601_CY, C1 = ITUR_BT_601_CVR, C2 = ITUR_BT_601_CVG, C3 = ITUR_BT_601_CUG,
              C4 = ITUR_BT_601_CUB;
    const int yuv4xx_shift = ITUR_BT_601_SHIFT;

    const int yy = max_func(0, Y - 16) * C0;
    const int uu = U - 128;
    const int vv = V - 128;

    r = static_cast<uint8_t>(clamp_func(cv_descale_func((yy + C1 * vv), yuv4xx_shift), 0, 255));
    g = static_cast<uint8_t>(clamp_func(cv_descale_func((yy + C2 * vv + C3 * uu), yuv4xx_shift), 0, 255));
    b = static_cast<uint8_t>(clamp_func(cv_descale_func((yy + C4 * uu), yuv4xx_shift), 0, 255));
}

__device__ __forceinline__ void yuv42xxp_to_rgb_kernel_full_range(const int& Y, const int& U, const int& V,
                                                                  uint8_t& r, uint8_t& g, uint8_t& b) {
    const float C0 = 1048230.1882352941f;  // == 1220542 / 255 * (235 - 16)
    const float C1 = 1470078.6196078432f;  // == 1673527 / 255 * (240 - 16)
    const float C2 = -748855.7176470588f;  // == -852492 / 255 * (240 - 16)
    const float C3 = -360150.7137254902f;  // == -409993 / 255 * (240 - 16)
    const float C4 = 1858783.6235294119f;  // == 2116026 / 255 * (240 - 16)
    const int32_t yuv4xx_shift = 20;

    const float yy = max_func(0.0f, static_cast<float>(Y)) * C0;

    const float uu = static_cast<float>(U) - 127.5f;
    const float vv = static_cast<float>(V) - 127.5f;

    const int32_t r_temp = static_cast<int32_t>(yy + C1 * vv + 0.5f);
    const int32_t g_temp = static_cast<int32_t>(yy + C2 * vv + C3 * uu + 0.5f);
    const int32_t b_temp = static_cast<int32_t>(yy + C4 * uu + 0.5f);

    r = static_cast<uint8_t>(clamp_func(cv_descale_func(r_temp, yuv4xx_shift), 0, 255));
    g = static_cast<uint8_t>(clamp_func(cv_descale_func(g_temp, yuv4xx_shift), 0, 255));
    b = static_cast<uint8_t>(clamp_func(cv_descale_func(b_temp, yuv4xx_shift), 0, 255));
}

static __global__ void nv12_to_rgb_kernel(uint8_t* y_data, uint8_t* uv_data, int2 size, int stride_y_dir_y_in,
                                          int stride_y_dir_uv_in, int stride_y_dir_out, int uidx,
                                          uint8_t* res_data, bool is_full_range, bool asBGR) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= size.x || y >= size.y) {
        return;
    }
    const int y_chroma = y / 2;
    const int x_uv = x - x % 2;  //(x % 2 == 0) ? x : (x - 1);
    const int x_u = x_uv + uidx;
    const int x_v = x_uv + 1 - uidx;
    const uint8_t y_val = y_data[y * stride_y_dir_y_in + x];
    const int row_chroma = y_chroma * stride_y_dir_uv_in;
    const uint8_t u_val = uv_data[row_chroma + x_u];
    const uint8_t v_val = uv_data[row_chroma + x_v];

    uint8_t r{0};
    uint8_t g{0};
    uint8_t b{0};
    if (is_full_range) {
        yuv42xxp_to_rgb_kernel_full_range(y_val, u_val, v_val, r, g, b);
    } else {
        yuv42xxp_to_rgb_kernel(y_val, u_val, v_val, r, g, b);
    }

    uint8_t* res_data_pixel = res_data + (y * stride_y_dir_out + x * 3);
    res_data_pixel[1] = g;
    if (asBGR) {
        res_data_pixel[0] = b;
        res_data_pixel[2] = r;
    } else {
        res_data_pixel[0] = r;
        res_data_pixel[2] = b;
    }
}

void convert_nv12_to_rgb(const CAIMemoryView& input_y, const CAIMemoryView& input_uv, RGBFrame& output,
                         bool is_full_range, bool asBGR) {
    dim3 block_size{32, 32, 1};

    // x is the second dimension and y the first in the CAIMemoryView. Hence,
    // input_y.shape[1] is used for x and input_y.shape[0] for y
    dim3 num_blocks = {(input_y.shape[1] + 31) / 32, (input_y.shape[0] + 31) / 32, 1};

    uint8_t* y_data = reinterpret_cast<uint8_t*>(input_y.data);

    uint8_t* uv_data = reinterpret_cast<uint8_t*>(input_uv.data);

    // x is the second dimension and y the first in the CAIMemoryView. Hence,
    // input_y.shape[1] is used for x and input_y.shape[0] for y
    const int2 size{input_y.shape[1], input_y.shape[0]};

    uint8_t* res_data = reinterpret_cast<uint8_t*>(output.data);

    // y is the first dimension in the CAIMemoryView. Hence, stride[0] is the
    // stride in
    nv12_to_rgb_kernel<<<num_blocks, block_size, 0, input_y.stream>>>(
        y_data, uv_data, size, input_y.stride[0], input_uv.stride[0], std::get<0>(output.stride), 0, res_data,
        is_full_range, asBGR);
    const cudaError last_error = cudaGetLastError();
    if (last_error != cudaError::cudaSuccess) {
        throw std::runtime_error("Color convertion kernel encountered error; Error code: " +
                                 std::to_string(last_error));
    }
    output.isBGR = asBGR;
}