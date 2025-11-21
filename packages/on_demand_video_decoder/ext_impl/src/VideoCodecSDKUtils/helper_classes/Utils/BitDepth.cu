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

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

static __global__ void ConvertUInt8ToUInt16Kernel(uint8_t *dpUInt8, uint16_t *dpUInt16, int nSrcPitch, int nDestPitch, int nWidth, int nHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nWidth || y >= nHeight)
    {
        return;
    }
    int destStrideInPixels = nDestPitch / (sizeof(uint16_t));
    *(uchar2 *)&dpUInt16[y * destStrideInPixels + x] = uchar2{ 0, dpUInt8[y * nSrcPitch + x] };
}

static __global__ void ConvertUInt16ToUInt8Kernel(uint16_t *dpUInt16, uint8_t *dpUInt8, int nSrcPitch, int nDestPitch, int nWidth, int nHeight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= nWidth || y >= nHeight)
    {
        return;
    }
    int srcStrideInPixels = nSrcPitch / (sizeof(uint16_t));
    dpUInt8[y * nDestPitch + x] = ((uchar2 *)&dpUInt16[y * srcStrideInPixels + x])->y;
}

void ConvertUInt8ToUInt16(uint8_t *dpUInt8, uint16_t *dpUInt16, int nSrcPitch, int nDestPitch, int nWidth, int nHeight)
{
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(((uint32_t)nWidth + blockSize.x - 1) / blockSize.x, ((uint32_t)nHeight + blockSize.y - 1) / blockSize.y, 1);
    ConvertUInt8ToUInt16Kernel <<< gridSize, blockSize >>>(dpUInt8, dpUInt16, nSrcPitch, nDestPitch, nWidth, nHeight);
}

void ConvertUInt16ToUInt8(uint16_t *dpUInt16, uint8_t *dpUInt8, int nSrcPitch, int nDestPitch, int nWidth, int nHeight)
{
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(((uint32_t)nWidth + blockSize.x - 1) / blockSize.x, ((uint32_t)nHeight + blockSize.y - 1) / blockSize.y, 1);
    ConvertUInt16ToUInt8Kernel <<<gridSize, blockSize >>>(dpUInt16, dpUInt8, nSrcPitch, nDestPitch, nWidth, nHeight);
}
