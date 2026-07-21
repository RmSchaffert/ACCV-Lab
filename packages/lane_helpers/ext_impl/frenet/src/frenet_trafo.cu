#include "frenet_trafo.cuh"

#include <cstddef>
#include <mutex>

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include "frenet_trafo_kernels.cuh"
#include "helper_macros.cuh"

namespace frenet {

static constexpr int MAX_CACHED_CUDA_DEVICES = 64;

using lane_helpers::ext_impl::check_non_negative_cuda_device;

static size_t query_default_shared_mem_for_device(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    return static_cast<size_t>(prop.sharedMemPerBlock);
}

static size_t default_shared_mem_for_device(int device) {
    static std::once_flag configured_devices[MAX_CACHED_CUDA_DEVICES];
    static size_t default_shared_mem_by_device[MAX_CACHED_CUDA_DEVICES] = {};

    check_non_negative_cuda_device(device);
    if (device >= MAX_CACHED_CUDA_DEVICES) {
        return query_default_shared_mem_for_device(device);
    }

    std::call_once(configured_devices[device], [device]() {
        default_shared_mem_by_device[device] = query_default_shared_mem_for_device(device);
    });
    return default_shared_mem_by_device[device];
}

void frenet_trafo_cuda(const at::Tensor& reference_lane_points, const at::Tensor& points_to_transform,
                       const at::Tensor* normals, bool normals_are_per_vertex,
                       at::Tensor& frenet_points_result) {
    const int num_ref_points = static_cast<int>(reference_lane_points.size(0));
    const int num_points_to_transform = static_cast<int>(points_to_transform.size(0));
    if (num_points_to_transform == 0) {
        return;
    }

    const size_t max_default_shared_mem = default_shared_mem_for_device(reference_lane_points.get_device());

    constexpr int block_size = 256;
    const dim3 block_dim(block_size, 1, 1);
    const dim3 grid_dim((num_points_to_transform + block_size - 1) / block_size, 1, 1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, reference_lane_points.scalar_type(), "frenet_trafo_cuda", [&] {
            const int num_segments = num_ref_points > 0 ? num_ref_points - 1 : 0;
            const size_t precomputed_shared_mem_size =
                static_cast<size_t>(num_segments) * sizeof(SharedReferenceSegment<scalar_t>);
            const size_t raw_points_shared_mem_size =
                static_cast<size_t>(num_ref_points) * 2 * sizeof(scalar_t);

            if (precomputed_shared_mem_size <= max_default_shared_mem) {
                frenet_trafo_kernel<scalar_t, SharedReferenceMode::PRECOMPUTED_SEGMENTS>
                    <<<grid_dim, block_dim, precomputed_shared_mem_size, stream>>>(
                        reference_lane_points.data_ptr<scalar_t>(), points_to_transform.data_ptr<scalar_t>(),
                        normals != nullptr ? normals->data_ptr<scalar_t>() : nullptr,
                        frenet_points_result.data_ptr<scalar_t>(), num_ref_points, num_points_to_transform,
                        normals_are_per_vertex);
            } else if (raw_points_shared_mem_size <= max_default_shared_mem) {
                frenet_trafo_kernel<scalar_t, SharedReferenceMode::RAW_POINTS>
                    <<<grid_dim, block_dim, raw_points_shared_mem_size, stream>>>(
                        reference_lane_points.data_ptr<scalar_t>(), points_to_transform.data_ptr<scalar_t>(),
                        normals != nullptr ? normals->data_ptr<scalar_t>() : nullptr,
                        frenet_points_result.data_ptr<scalar_t>(), num_ref_points, num_points_to_transform,
                        normals_are_per_vertex);
            } else {
                frenet_trafo_kernel<scalar_t, SharedReferenceMode::NONE><<<grid_dim, block_dim, 0, stream>>>(
                    reference_lane_points.data_ptr<scalar_t>(), points_to_transform.data_ptr<scalar_t>(),
                    normals != nullptr ? normals->data_ptr<scalar_t>() : nullptr,
                    frenet_points_result.data_ptr<scalar_t>(), num_ref_points, num_points_to_transform,
                    normals_are_per_vertex);
            }
            CUDA_CHECK_LAST();
        });
}

}  // namespace frenet