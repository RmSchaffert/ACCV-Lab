#include <pybind11/pybind11.h>
#include <cstdint>

#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include "frenet_trafo.cuh"
#include "input_checks.h"

namespace py = pybind11;

namespace frenet {

    using lane_helpers::ext_impl::check_same_device;
    using lane_helpers::ext_impl::check_same_dtype;

    static at::Tensor validate_normals_tensor(
        const at::Tensor& reference_lane_points, const py::object& normals_object,
        int expected_num_normals, const char* normals_name, const char* expected_shape) {
        const at::Tensor normals = normals_object.cast<at::Tensor>();
        CHECK_DEVICE(normals);
        CHECK_TYPE(normals);
        AT_ASSERTM(normals.ndimension() == 2, normals_name, " must have shape ", expected_shape);
        AT_ASSERTM(normals.size(0) == expected_num_normals, normals_name,
                   " must have shape ", expected_shape);
        AT_ASSERTM(normals.size(1) == 2, normals_name, " must have shape ", expected_shape);
        check_same_dtype(reference_lane_points, normals,
                         "reference_lane_points and normals must have the same dtype");
        check_same_device(reference_lane_points, normals,
                          "reference_lane_points and normals must be on the same device");
        return normals.contiguous();
    }

    at::Tensor frenet_trafo(
        at::Tensor reference_lane_points, at::Tensor points_to_transform,
        py::object per_segment_normals, py::object per_vertex_normals) {
        CHECK_DEVICE(reference_lane_points);
        CHECK_DEVICE(points_to_transform);
        CHECK_TYPE(reference_lane_points);
        CHECK_TYPE(points_to_transform);
        AT_ASSERTM(reference_lane_points.ndimension() == 2, "reference_lane_points must have shape (num_points, 2)");
        AT_ASSERTM(points_to_transform.ndimension() == 2, "points_to_transform must have shape (num_points, 2)");
        AT_ASSERTM(reference_lane_points.size(1) == 2, "reference_lane_points must have shape (num_points, 2)");
        AT_ASSERTM(points_to_transform.size(1) == 2, "points_to_transform must have shape (num_points, 2)");
        check_same_dtype(reference_lane_points, points_to_transform,
                         "reference_lane_points and points_to_transform must have the same dtype");
        check_same_device(reference_lane_points, points_to_transform, "reference_lane_points and points_to_transform must be on the same device");

        const bool has_per_segment_normals = !per_segment_normals.is_none();
        const bool has_per_vertex_normals = !per_vertex_normals.is_none();
        TORCH_CHECK(!(has_per_segment_normals && has_per_vertex_normals),
                    "Provide either per_segment_normals or per_vertex_normals, not both");

        at::Tensor normals_contiguous;
        const at::Tensor* normals = nullptr;
        bool normals_are_per_vertex = false;
        if (has_per_segment_normals) {
            normals_contiguous = validate_normals_tensor(
                reference_lane_points, per_segment_normals,
                static_cast<int>(reference_lane_points.size(0) - 1), "per_segment_normals",
                "(num_reference_points - 1, 2)");
            normals = &normals_contiguous;
        } else if (has_per_vertex_normals) {
            normals_contiguous = validate_normals_tensor(
                reference_lane_points, per_vertex_normals,
                static_cast<int>(reference_lane_points.size(0)), "per_vertex_normals",
                "(num_reference_points, 2)");
            normals = &normals_contiguous;
            normals_are_per_vertex = true;
        }

        const at::Tensor reference_lane_points_contiguous = reference_lane_points.contiguous();
        const at::Tensor points_to_transform_contiguous = points_to_transform.contiguous();
        at::Tensor frenet_points = at::empty_like(points_to_transform_contiguous);

        if (reference_lane_points.is_cuda()) {
            frenet_trafo_cuda(reference_lane_points_contiguous, points_to_transform_contiguous,
                              normals, normals_are_per_vertex, frenet_points);
        } else {
            frenet_trafo_cpu(reference_lane_points_contiguous, points_to_transform_contiguous,
                             normals, normals_are_per_vertex, frenet_points);
        }

        return frenet_points;

    }

}

using namespace frenet;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Lane helpers frenet bindings";
    m.def("frenet_trafo", &frenet_trafo,
          py::arg("reference_lane_points"), py::arg("points_to_transform"),
          py::arg("per_segment_normals") = py::none(),
          py::arg("per_vertex_normals") = py::none(),
          "Internal entry point for transforming points to frenet coordinates.");
}