#include "frenet_trafo.cuh"

#include <limits>
#include <vector>

#include <ATen/AccumulateType.h>
#include <ATen/Parallel.h>
#include <torch/torch.h>

#include "frenet_trafo_common.cuh"

namespace frenet {

template <typename dtype>
using cpu_acc_t = at::acc_type<dtype, false>;

template <typename dtype, typename accum_dtype>
struct PrecomputedReferenceSegment {
    accum_dtype s_start;
    Vector2D<dtype> tangent;
    Vector2D<dtype> normalized_tangent;
    dtype length;
    bool is_valid;
};

template <typename dtype, typename accum_dtype>
static void precompute_reference_segments(
    const VectorView<dtype>& reference_lane, int num_segments,
    std::vector<PrecomputedReferenceSegment<dtype, accum_dtype>>& segments) {
    accum_dtype segment_start_s = static_cast<accum_dtype>(0.0);
    for (int segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
        // Get the points of the current segment and the tangent of the segment.
        const VectorView<dtype> segment_points = reference_lane.get_advanced_view(segment_idx);
        Vector2D<dtype> point1;
        Vector2D<dtype> point2;
        segment_points.get_first_two(point1, point2);

        PrecomputedReferenceSegment<dtype, accum_dtype>& segment = segments[segment_idx];
        segment.s_start = segment_start_s;
        segment.tangent = get_tangent(point1, point2);

        // Precompute reusable segment geometry once for all query points.
        const dtype length_squared = segment.tangent.length_squared();
        if (value_gt(length_squared, static_cast<dtype>(0.0))) {
            // Normalize the tangent, store the segment length, and update the accumulated s-coordinate.
            segment.length = sqrt_compat(length_squared);
            const dtype inv_length = static_cast<dtype>(1.0) / segment.length;
            segment.normalized_tangent = segment.tangent * inv_length;
            segment.is_valid = true;
            segment_start_s += static_cast<accum_dtype>(segment.length);
        } else {
            // If the segment is invalid (length of 0), mark it so query-point processing can skip it.
            segment.normalized_tangent = {static_cast<dtype>(0.0), static_cast<dtype>(0.0)};
            segment.length = static_cast<dtype>(0.0);
            segment.is_valid = false;
        }
    }
}

template <typename dtype, typename accum_dtype>
static Vector2D<dtype> get_projection_normal_precomputed(
    const std::vector<PrecomputedReferenceSegment<dtype, accum_dtype>>& segments,
    int segment_idx, int num_segments, ProjResType proj_res_type,
    const Vector2D<dtype>& segment_unit_normal) {
    // Use precomputed neighboring tangents to average normals at interior joints.
    if (proj_res_type == ProjResType::BELOW_START && segment_idx > 0) {
        const PrecomputedReferenceSegment<dtype, accum_dtype>& prev_segment =
            segments[segment_idx - 1];
        const Vector2D<dtype> prev_normal =
            prev_segment.is_valid
                ? get_normal_from_tangent(prev_segment.normalized_tangent)
                : Vector2D<dtype>(static_cast<dtype>(0.0), static_cast<dtype>(0.0));
        return normalize_or_fallback(prev_normal + segment_unit_normal, segment_unit_normal);
    }
    if (proj_res_type == ProjResType::ABOVE_END && segment_idx + 1 < num_segments) {
        const PrecomputedReferenceSegment<dtype, accum_dtype>& next_segment =
            segments[segment_idx + 1];
        const Vector2D<dtype> next_normal =
            next_segment.is_valid
                ? get_normal_from_tangent(next_segment.normalized_tangent)
                : Vector2D<dtype>(static_cast<dtype>(0.0), static_cast<dtype>(0.0));
        return normalize_or_fallback(segment_unit_normal + next_normal, segment_unit_normal);
    }
    return segment_unit_normal;
}

template <typename dtype, typename accum_dtype>
static void transform_point_precomputed(
    const VectorView<dtype>& reference_lane, const Vector2D<dtype>& point, dtype* frenet_point,
    int num_ref_points, const dtype* normals, bool normals_are_per_vertex,
    const std::vector<PrecomputedReferenceSegment<dtype, accum_dtype>>& segments) {
    if (num_ref_points < 2) {
        const dtype nan_value = std::numeric_limits<dtype>::quiet_NaN();
        frenet_point[0] = nan_value;
        frenet_point[1] = nan_value;
        return;
    }

    const int num_segments = num_ref_points - 1;

    // Track the closest projection using the precomputed start s-coordinate per segment.
    Vector2D<dtype> best_projection;
    dtype best_distance_squared = std::numeric_limits<dtype>::infinity();
    accum_dtype best_s = static_cast<accum_dtype>(0.0);
    int best_segment_idx = -1;
    ProjResType best_proj_res_type = ProjResType::INSIDE_SEGMENT;
    dtype best_segment_length = static_cast<dtype>(1.0);
    dtype best_projection_length = static_cast<dtype>(0.0);

    for (int segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
        const PrecomputedReferenceSegment<dtype, accum_dtype>& segment = segments[segment_idx];
        // If the segment is invalid (length of 0), skip it.
        if (!segment.is_valid) {
            continue;
        }

        // Get the points of the current segment; length and normalized tangent are precomputed.
        const VectorView<dtype> segment_points = reference_lane.get_advanced_view(segment_idx);
        Vector2D<dtype> point1;
        Vector2D<dtype> point2;
        segment_points.get_first_two(point1, point2);

        // Project onto the current segment, including endpoint clamping.
        Vector2D<dtype> projection;
        dtype distance_squared = static_cast<dtype>(0.0);
        ProjResType proj_res_type = ProjResType::INSIDE_SEGMENT;
        get_projection_point_distance_and_is_inside(
            point1, point2, segment.normalized_tangent, segment.length, point, projection,
            distance_squared, proj_res_type);

        // Update the best projection together with the matching clamped s-coordinate.
        if (value_lt(distance_squared, best_distance_squared)) {
            best_projection = projection;
            best_distance_squared = distance_squared;
            best_segment_idx = segment_idx;
            best_proj_res_type = proj_res_type;
            best_segment_length = segment.length;

            if (proj_res_type == ProjResType::BELOW_START) {
                best_s = segment.s_start;
                best_projection_length = static_cast<dtype>(0.0);
            } else if (proj_res_type == ProjResType::ABOVE_END) {
                best_s = segment.s_start + static_cast<accum_dtype>(segment.length);
                best_projection_length = segment.length;
            } else {
                const dtype projection_length = (projection - point1).dot(segment.normalized_tangent);
                best_s = segment.s_start + static_cast<accum_dtype>(projection_length);
                best_projection_length = projection_length;
            }
        }
    }

    // If no valid segment was found (all segments have length 0), return NaN.
    if (best_segment_idx < 0) {
        const dtype nan_value = std::numeric_limits<dtype>::quiet_NaN();
        frenet_point[0] = nan_value;
        frenet_point[1] = nan_value;
        return;
    }

    const PrecomputedReferenceSegment<dtype, accum_dtype>& best_segment =
        segments[best_segment_idx];
    // Select the signed-distance normal from geometry, segment normals, or interpolated vertex normals.
    Vector2D<dtype> best_normal;
    if (normals == nullptr) {
        const Vector2D<dtype> best_segment_normal =
            get_normal_from_tangent(best_segment.normalized_tangent);
        best_normal = get_projection_normal_precomputed(
            segments, best_segment_idx, num_segments, best_proj_res_type, best_segment_normal);
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

template <typename dtype>
static void frenet_trafo_cpu_impl(const dtype* reference_lane_points,
                                  const dtype* points_to_transform,
                                  const dtype* normals, bool normals_are_per_vertex,
                                  dtype* frenet_points_result, int num_ref_points,
                                  int num_points_to_transform) {
    using accum_dtype = cpu_acc_t<dtype>;
    const VectorView<dtype> reference_lane(reference_lane_points);
    const int num_segments = num_ref_points > 0 ? num_ref_points - 1 : 0;
    std::vector<PrecomputedReferenceSegment<dtype, accum_dtype>> segments(num_segments);
    // Precompute the reference lane once; each worker then reuses the immutable segment data.
    precompute_reference_segments<dtype, accum_dtype>(reference_lane, num_segments, segments);

    at::parallel_for(0, num_points_to_transform, 0, [&](int64_t start, int64_t end) {
        for (int64_t point_idx = start; point_idx < end; ++point_idx) {
            const Vector2D<dtype> point(points_to_transform + point_idx * 2);
            transform_point_precomputed<dtype, accum_dtype>(
                reference_lane, point, frenet_points_result + point_idx * 2, num_ref_points,
                normals, normals_are_per_vertex,
                segments);
        }
    });
}

void frenet_trafo_cpu(const at::Tensor& reference_lane_points,
                      const at::Tensor& points_to_transform,
                      const at::Tensor* normals, bool normals_are_per_vertex,
                      at::Tensor& frenet_points_result) {
    const int num_ref_points = static_cast<int>(reference_lane_points.size(0));
    const int num_points_to_transform = static_cast<int>(points_to_transform.size(0));
    if (num_points_to_transform == 0) {
        return;
    }

    AT_DISPATCH_FLOATING_TYPES(reference_lane_points.scalar_type(), "frenet_trafo_cpu", [&] {
        frenet_trafo_cpu_impl<scalar_t>(
            reference_lane_points.data_ptr<scalar_t>(),
            points_to_transform.data_ptr<scalar_t>(),
            normals != nullptr ? normals->data_ptr<scalar_t>() : nullptr,
            normals_are_per_vertex,
            frenet_points_result.data_ptr<scalar_t>(), num_ref_points, num_points_to_transform);
    });
}

}  // namespace frenet
