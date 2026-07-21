#ifndef LANE_HELPERS_FRENET_TRAFO_COMMON_CUH
#define LANE_HELPERS_FRENET_TRAFO_COMMON_CUH

#include "dtype_compat.cuh"

#ifdef __CUDACC__
#define LANE_HELPERS_FRENET_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define LANE_HELPERS_FRENET_HOST_DEVICE_INLINE inline
#endif

namespace frenet {

#ifdef __CUDACC__
using lane_helpers::ext_impl::rsqrt_compat;
#endif
using lane_helpers::ext_impl::sqrt_compat;
using lane_helpers::ext_impl::value_gt;
using lane_helpers::ext_impl::value_lt;

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE dtype inv_sqrt_for_normalization(dtype value) {
#ifdef __CUDA_ARCH__
    return rsqrt_compat(value);
#else
    return static_cast<dtype>(1.0) / sqrt_compat(value);
#endif
}

template <typename dtype>
struct Vector2D {
    dtype x;
    dtype y;

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D() = default;

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D(dtype x, dtype y) : x(x), y(y) {}

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE explicit Vector2D(const dtype* point)
        : x(point[0]), y(point[1]) {}

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE void write_back(dtype* point) const {
        point[0] = x;
        point[1] = y;
    }

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE dtype length() const {
        return sqrt_compat(length_squared());
    }

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE dtype inv_length() const {
        return inv_sqrt_for_normalization(length_squared());
    }

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE dtype length_squared() const {
        return x * x + y * y;
    }

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE dtype dot(const Vector2D& other) const {
        return x * other.x + y * other.y;
    }
};

template <typename dtype>
struct VectorView {
    const dtype* data;

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE explicit VectorView(const dtype* data) : data(data) {}

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> operator[](int index) const {
        return Vector2D<dtype>(data + index * 2);
    }

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE VectorView get_advanced_view(int offset) const {
        return VectorView(data + offset * 2);
    }

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> get_first() const {
        return Vector2D<dtype>(data);
    }

    LANE_HELPERS_FRENET_HOST_DEVICE_INLINE void get_first_two(
        Vector2D<dtype>& out1, Vector2D<dtype>& out2) const {
        out1 = Vector2D<dtype>(data);
        out2 = Vector2D<dtype>(data + 2);
    }
};

enum struct ProjResType {
    INSIDE_SEGMENT = 0,
    BELOW_START = 1,
    ABOVE_END = 2,
};

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> operator+(
    const Vector2D<dtype>& a, const Vector2D<dtype>& b) {
    return {a.x + b.x, a.y + b.y};
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> operator-(
    const Vector2D<dtype>& a, const Vector2D<dtype>& b) {
    return {a.x - b.x, a.y - b.y};
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> operator*(
    const Vector2D<dtype>& a, dtype b) {
    return {a.x * b, a.y * b};
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> get_normal(
    const Vector2D<dtype>& point1, const Vector2D<dtype>& point2) {
    return {point1.y - point2.y, point2.x - point1.x};
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> get_tangent(
    const Vector2D<dtype>& point1, const Vector2D<dtype>& point2) {
    return {point2.x - point1.x, point2.y - point1.y};
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> get_normal_from_tangent(
    const Vector2D<dtype>& tangent) {
    return {-tangent.y, tangent.x};
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> get_segment_unit_normal(
    const VectorView<dtype>& segment_points) {
    Vector2D<dtype> point1;
    Vector2D<dtype> point2;
    segment_points.get_first_two(point1, point2);
    const Vector2D<dtype> normal = get_normal(point1, point2);
    const dtype normal_length_squared = normal.length_squared();
    if (value_gt(normal_length_squared, static_cast<dtype>(0.0))) {
        const Vector2D<dtype> normalized_normal = normal * inv_sqrt_for_normalization(normal_length_squared);
        return normalized_normal;
    }
    return {static_cast<dtype>(0.0), static_cast<dtype>(0.0)};
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> normalize_or_fallback(
    const Vector2D<dtype>& vector, const Vector2D<dtype>& fallback) {
    const dtype vector_length_squared = vector.length_squared();
    if (value_gt(vector_length_squared, static_cast<dtype>(0.0))) {
        const Vector2D<dtype> normalized_vector = vector * inv_sqrt_for_normalization(vector_length_squared);
        return normalized_vector;
    }
    return fallback;
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> get_projection_normal(
    const VectorView<dtype>& reference_lane_points, int segment_idx, int num_segments,
    ProjResType proj_res_type, const Vector2D<dtype>& segment_unit_normal) {
    // Use averaged normals for clamped projections at interior reference-lane joints.
    if (proj_res_type == ProjResType::BELOW_START && segment_idx > 0) {
        const Vector2D<dtype> prev_normal =
            get_segment_unit_normal<dtype>(reference_lane_points.get_advanced_view(segment_idx - 1));
        const Vector2D<dtype> averaged_normal = normalize_or_fallback(prev_normal + segment_unit_normal, segment_unit_normal);
        return averaged_normal;
    }
    if (proj_res_type == ProjResType::ABOVE_END && segment_idx + 1 < num_segments) {
        const Vector2D<dtype> next_normal =
            get_segment_unit_normal<dtype>(reference_lane_points.get_advanced_view(segment_idx + 1));
        const Vector2D<dtype> averaged_normal = normalize_or_fallback(segment_unit_normal + next_normal, segment_unit_normal);
        return averaged_normal;
    }
    // If the projection is inside a segment or at the beginning/end of the reference lane, use the segment normal.
    return segment_unit_normal;
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> get_projection_normal_from_reference_geometry(
    const VectorView<dtype>& reference_lane_points, int segment_idx, int num_segments,
    ProjResType proj_res_type, dtype segment_length) {
    const VectorView<dtype> segment_points = reference_lane_points.get_advanced_view(segment_idx);
    Vector2D<dtype> point1;
    Vector2D<dtype> point2;
    segment_points.get_first_two(point1, point2);

    const Vector2D<dtype> segment_normal =
        get_normal(point1, point2) * (static_cast<dtype>(1.0) / segment_length);
    const Vector2D<dtype> projection_normal = get_projection_normal<dtype>(
        reference_lane_points, segment_idx, num_segments, proj_res_type, segment_normal);
    return projection_normal;
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> get_projection_normal_from_segment_normals(
    const VectorView<dtype>& segment_normals, int segment_idx, int num_segments,
    ProjResType proj_res_type) {
    const Vector2D<dtype> segment_normal = segment_normals[segment_idx];
    if (proj_res_type == ProjResType::BELOW_START && segment_idx > 0) {
        const Vector2D<dtype> averaged_normal = normalize_or_fallback(segment_normals[segment_idx - 1] + segment_normal, segment_normal);
        return averaged_normal;
    }
    if (proj_res_type == ProjResType::ABOVE_END && segment_idx + 1 < num_segments) {
        const Vector2D<dtype> averaged_normal = normalize_or_fallback(segment_normal + segment_normals[segment_idx + 1], segment_normal);
        return averaged_normal;
    }
    return segment_normal;
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE Vector2D<dtype> get_projection_normal_from_vertex_normals(
    const VectorView<dtype>& vertex_normals, int segment_idx, dtype projection_fraction) {
    const Vector2D<dtype> normal_start = vertex_normals[segment_idx];
    const Vector2D<dtype> normal_end = vertex_normals[segment_idx + 1];
    const Vector2D<dtype> interpolated_normal =
        normal_start * (static_cast<dtype>(1.0) - projection_fraction) +
        normal_end * projection_fraction;
    const Vector2D<dtype> averaged_normal = normalize_or_fallback(interpolated_normal, interpolated_normal);
    return averaged_normal;
}

template <typename dtype>
LANE_HELPERS_FRENET_HOST_DEVICE_INLINE void get_projection_point_distance_and_is_inside(
    const Vector2D<dtype>& point1, const Vector2D<dtype>& point2,
    const Vector2D<dtype>& normalized_tangent, dtype segment_length,
    const Vector2D<dtype>& to_proj, Vector2D<dtype>& projection_point, dtype& distance_squared,
    ProjResType& proj_res_type) {
    const Vector2D<dtype> local_pos_to_proj = to_proj - point1;

    // Project into the segment coordinate and clamp to the closest valid segment point.
    const dtype projection_length = local_pos_to_proj.dot(normalized_tangent);
    const bool is_below_start = value_lt(projection_length, static_cast<dtype>(0.0));
    const bool is_above_end = value_gt(projection_length, segment_length);

    //If the projeciton of the point is outside the segment, the shortest path to the segment is the distance to the closest 
    // segment point, not the distance of the point to its projection.
    if (is_below_start) {
        projection_point = point1;
        distance_squared = local_pos_to_proj.length_squared();
        proj_res_type = ProjResType::BELOW_START;
    } else if (is_above_end) {
        projection_point = point2;
        distance_squared = (to_proj - point2).length_squared();
        proj_res_type = ProjResType::ABOVE_END;
    } else {
        projection_point = point1 + normalized_tangent * projection_length;
        distance_squared = (to_proj - projection_point).length_squared();
        proj_res_type = ProjResType::INSIDE_SEGMENT;
    }
}

}  // namespace frenet

#undef LANE_HELPERS_FRENET_HOST_DEVICE_INLINE

#endif  // LANE_HELPERS_FRENET_TRAFO_COMMON_CUH
