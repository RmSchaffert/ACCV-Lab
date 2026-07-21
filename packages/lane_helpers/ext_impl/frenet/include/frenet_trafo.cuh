#ifndef LANE_HELPERS_FRENET_TRAFO_CUH
#define LANE_HELPERS_FRENET_TRAFO_CUH

namespace at {
class Tensor;
}

namespace frenet {

    void frenet_trafo_cuda(const at::Tensor& reference_lane_points, const at::Tensor& points_to_transform,
                           const at::Tensor* normals, bool normals_are_per_vertex,
                           at::Tensor& frenet_points_result);
    void frenet_trafo_cpu(const at::Tensor& reference_lane_points, const at::Tensor& points_to_transform,
                          const at::Tensor* normals, bool normals_are_per_vertex,
                          at::Tensor& frenet_points_result);

}

#endif  // LANE_HELPERS_FRENET_TRAFO_CUH