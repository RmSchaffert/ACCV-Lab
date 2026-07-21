#ifndef LANE_HELPERS_FRENET_KERNELS_CUH
#define LANE_HELPERS_FRENET_KERNELS_CUH

#include <ATen/AccumulateType.h>

#include "frenet_trafo_common.cuh"

namespace frenet {

enum struct SharedReferenceMode {
    PRECOMPUTED_SEGMENTS,
    RAW_POINTS,
    NONE,
};

template <typename dtype>
struct SharedReferenceSegment {
    using accum_dtype = at::acc_type<dtype, true>;

    Vector2D<dtype> point1;
    Vector2D<dtype> point2;
    Vector2D<dtype> normalized_tangent;
    dtype length;
    accum_dtype s_start;
};

template <typename dtype>
__device__ __forceinline__ void precompute_reference_segments(
    const dtype* __restrict__ reference_lane_points, SharedReferenceSegment<dtype>* __restrict__ segments,
    int num_segments) {
    // Pre-compute  the segments
    for (int segment_idx = threadIdx.x; segment_idx < num_segments; segment_idx += blockDim.x) {
        SharedReferenceSegment<dtype>& segment = segments[segment_idx];
        segment.point1 = Vector2D<dtype>(reference_lane_points + segment_idx * 2);
        segment.point2 = Vector2D<dtype>(reference_lane_points + (segment_idx + 1) * 2);

        const Vector2D<dtype> tangent = get_tangent(segment.point1, segment.point2);
        const dtype length_squared = tangent.length_squared();
        if (value_gt(length_squared, static_cast<dtype>(0.0))) {
            const dtype inv_length = rsqrt_compat(length_squared);
            segment.normalized_tangent = tangent * inv_length;
            segment.length = static_cast<dtype>(1.0) / inv_length;
        } else {
            segment.normalized_tangent = {static_cast<dtype>(0.0), static_cast<dtype>(0.0)};
            segment.length = static_cast<dtype>(0.0);
        }
    }
    __syncthreads();

    // Accumulate the start s for the segments
    if (threadIdx.x == 0) {
        using accum_dtype = at::acc_type<dtype, true>;
        accum_dtype segment_start_s = static_cast<accum_dtype>(0.0);
        for (int segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
            SharedReferenceSegment<dtype>& segment = segments[segment_idx];
            segment.s_start = segment_start_s;
            segment_start_s += static_cast<accum_dtype>(segment.length);
        }
    }
    __syncthreads();
}

template <typename dtype>
__device__ __forceinline__ Vector2D<dtype> get_projection_normal_from_precomputed_geometry(
    const SharedReferenceSegment<dtype>* segments, int segment_idx, int num_segments,
    ProjResType proj_res_type) {
    const SharedReferenceSegment<dtype>& segment = segments[segment_idx];
    const Vector2D<dtype> segment_normal = get_normal_from_tangent(segment.normalized_tangent);

    if (proj_res_type == ProjResType::BELOW_START && segment_idx > 0) {
        const SharedReferenceSegment<dtype>& prev_segment = segments[segment_idx - 1];
        const Vector2D<dtype> prev_normal =
            value_gt(prev_segment.length, static_cast<dtype>(0.0))
                ? get_normal_from_tangent(prev_segment.normalized_tangent)
                : Vector2D<dtype>(static_cast<dtype>(0.0), static_cast<dtype>(0.0));
        return normalize_or_fallback(prev_normal + segment_normal, segment_normal);
    }
    if (proj_res_type == ProjResType::ABOVE_END && segment_idx + 1 < num_segments) {
        const SharedReferenceSegment<dtype>& next_segment = segments[segment_idx + 1];
        const Vector2D<dtype> next_normal =
            value_gt(next_segment.length, static_cast<dtype>(0.0))
                ? get_normal_from_tangent(next_segment.normalized_tangent)
                : Vector2D<dtype>(static_cast<dtype>(0.0), static_cast<dtype>(0.0));
        return normalize_or_fallback(segment_normal + next_normal, segment_normal);
    }
    return segment_normal;
}

template <typename dtype, bool use_precomputed_segments>
__device__ __forceinline__ void transform_point(const VectorView<dtype>& reference_lane,
                                                const SharedReferenceSegment<dtype>* precomputed_segments,
                                                const dtype* __restrict__ normals,
                                                bool normals_are_per_vertex, const Vector2D<dtype>& point,
                                                dtype* __restrict__ frenet_point, int num_ref_points) {
    using accum_dtype = at::acc_type<dtype, true>;

    if (num_ref_points < 2) {
        const dtype nan_value = static_cast<dtype>(NAN);
        frenet_point[0] = nan_value;
        frenet_point[1] = nan_value;
        return;
    }

    const int num_segments = num_ref_points - 1;

    // Track the closest projection while accumulating the start s-coordinate per segment.
    Vector2D<dtype> best_projection;
    dtype best_distance_squared = static_cast<dtype>(INFINITY);
    accum_dtype best_s = static_cast<accum_dtype>(0.0);
    int best_segment_idx = -1;
    ProjResType best_proj_res_type = ProjResType::INSIDE_SEGMENT;
    dtype best_segment_length = static_cast<dtype>(1.0);
    dtype best_projection_length = static_cast<dtype>(0.0);

    accum_dtype segment_start_s = static_cast<accum_dtype>(0.0);
    for (int segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
        Vector2D<dtype> point1;
        Vector2D<dtype> point2;
        Vector2D<dtype> normalized_tangent;
        dtype segment_length;
        accum_dtype current_segment_start_s;

        if constexpr (use_precomputed_segments) {
            const SharedReferenceSegment<dtype>& segment = precomputed_segments[segment_idx];
            point1 = segment.point1;
            point2 = segment.point2;
            normalized_tangent = segment.normalized_tangent;
            segment_length = segment.length;
            current_segment_start_s = segment.s_start;
            if (!value_gt(segment_length, static_cast<dtype>(0.0))) {
                continue;
            }
        } else {
            const VectorView<dtype> segment_points = reference_lane.get_advanced_view(segment_idx);
            segment_points.get_first_two(point1, point2);
            const Vector2D<dtype> tangent = get_tangent(point1, point2);
            const dtype segment_length_squared = tangent.length_squared();
            if (!value_gt(segment_length_squared, static_cast<dtype>(0.0))) {
                continue;
            }

            const dtype inv_segment_length = rsqrt_compat(segment_length_squared);
            normalized_tangent = tangent * inv_segment_length;
            segment_length = static_cast<dtype>(1.0) / inv_segment_length;
            current_segment_start_s = segment_start_s;
        }

        // Project onto the current segment, including endpoint clamping.
        Vector2D<dtype> projection;
        dtype distance_squared = static_cast<dtype>(0.0);
        ProjResType proj_res_type = ProjResType::INSIDE_SEGMENT;
        get_projection_point_distance_and_is_inside(point1, point2, normalized_tangent, segment_length, point,
                                                    projection, distance_squared, proj_res_type);

        // Keep the closest segment and the matching longitudinal Frenet coordinate (s-coordinate).
        if (value_lt(distance_squared, best_distance_squared)) {
            best_projection = projection;
            best_distance_squared = distance_squared;
            best_segment_idx = segment_idx;
            best_proj_res_type = proj_res_type;
            best_segment_length = segment_length;

            if (proj_res_type == ProjResType::BELOW_START) {
                best_s = current_segment_start_s;
                best_projection_length = static_cast<dtype>(0.0);
            } else if (proj_res_type == ProjResType::ABOVE_END) {
                best_s = current_segment_start_s + static_cast<accum_dtype>(segment_length);
                best_projection_length = segment_length;
            } else {
                const dtype projection_length = (projection - point1).dot(normalized_tangent);
                best_s = current_segment_start_s + static_cast<accum_dtype>(projection_length);
                best_projection_length = projection_length;
            }
        }

        if constexpr (!use_precomputed_segments) {
            segment_start_s += static_cast<accum_dtype>(segment_length);
        }
    }

    // If no valid segment was found (all segments have length 0), return NaN.
    if (best_segment_idx < 0) {
        const dtype nan_value = static_cast<dtype>(NAN);
        frenet_point[0] = nan_value;
        frenet_point[1] = nan_value;
        return;
    }

    // Select the signed-distance normal from geometry, segment normals, or interpolated vertex normals.
    Vector2D<dtype> best_normal;
    if (normals == nullptr) {
        if constexpr (use_precomputed_segments) {
            best_normal = get_projection_normal_from_precomputed_geometry(
                precomputed_segments, best_segment_idx, num_segments, best_proj_res_type);
        } else {
            best_normal = get_projection_normal_from_reference_geometry<dtype>(
                reference_lane, best_segment_idx, num_segments, best_proj_res_type, best_segment_length);
        }
    } else if (normals_are_per_vertex) {
        const dtype best_projection_fraction = best_projection_length / best_segment_length;
        best_normal = get_projection_normal_from_vertex_normals<dtype>(
            VectorView<dtype>(normals), best_segment_idx, best_projection_fraction);
    } else {
        best_normal = get_projection_normal_from_segment_normals<dtype>(
            VectorView<dtype>(normals), best_segment_idx, num_segments, best_proj_res_type);
    }
    const Vector2D<dtype> projection_to_point = point - best_projection;
    frenet_point[0] = static_cast<dtype>(best_s);
    frenet_point[1] = projection_to_point.dot(best_normal);
}

template <typename dtype, SharedReferenceMode shared_reference_mode>
__global__ void frenet_trafo_kernel(const dtype* __restrict__ reference_lane_points,
                                    const dtype* __restrict__ points_to_transform,
                                    const dtype* __restrict__ normals, dtype* __restrict__ frenet_points,
                                    int num_ref_points, int num_points_to_transform,
                                    bool normals_are_per_vertex) {
    extern __shared__ unsigned char shared_reference_raw[];

    const dtype* reference_lane_points_to_use = reference_lane_points;
    SharedReferenceSegment<dtype>* precomputed_segments = nullptr;
    if constexpr (shared_reference_mode == SharedReferenceMode::PRECOMPUTED_SEGMENTS) {
        precomputed_segments = reinterpret_cast<SharedReferenceSegment<dtype>*>(shared_reference_raw);
        const int num_segments = num_ref_points > 0 ? num_ref_points - 1 : 0;
        precompute_reference_segments(reference_lane_points, precomputed_segments, num_segments);
    } else if constexpr (shared_reference_mode == SharedReferenceMode::RAW_POINTS) {
        dtype* shared_reference_lane_points = reinterpret_cast<dtype*>(shared_reference_raw);
        const int num_ref_values = num_ref_points * 2;
        // Stage the reference lane once per block so all threads can reuse it.
        for (int i = threadIdx.x; i < num_ref_values; i += blockDim.x) {
            shared_reference_lane_points[i] = reference_lane_points[i];
        }
        __syncthreads();
        reference_lane_points_to_use = shared_reference_lane_points;
    }

    const int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points_to_transform) {
        return;
    }

    const VectorView<dtype> reference_lane(reference_lane_points_to_use);
    const Vector2D<dtype> point(points_to_transform + point_idx * 2);
    transform_point<dtype, shared_reference_mode == SharedReferenceMode::PRECOMPUTED_SEGMENTS>(
        reference_lane, precomputed_segments, normals, normals_are_per_vertex, point,
        frenet_points + point_idx * 2, num_ref_points);
}

}  // namespace frenet

#endif  // LANE_HELPERS_FRENET_KERNELS_CUH