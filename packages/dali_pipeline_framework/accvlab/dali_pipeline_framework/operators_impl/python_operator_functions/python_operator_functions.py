# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
import cupy

from nvidia.dali.pipeline import do_not_convert


@do_not_convert
def apply_transform_to_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    '''Apply an affine transformation to a set of points.

    Note that the transformation is applied in homogeneous coordinates.
    This is handled by the function and the input points are not expected to be in
    homogeneous coordinates. Similarly, the output points are not in homogeneous coordinates.

    Important:
        This function can handle concatenated points, i.e. multiple points per row, i.e.
        ``[x1, y1, x2, y2, ...]``.

    See also:
        A similar function is available in the :mod:`accvlab.dali_pipeline_framework.operators_impl.numba_operators`
        module. That function is faster and supports different variations of matrix multiplication (e.g.
        including use of transposed input and matrices, use of the inverse matrix, etc.),
        but it does not support concatenated points.

    Args:
        points: A 2D array of shape ``(N, 2 * num_points_per_row)`` containing the coordinates
            of the points.
        transform: A 2D array of shape ``(3, 3)`` containing the affine transformation matrix.
    Returns:
        A 2D array of shape ``(N, 2 * num_points_per_row)`` containing the coordinates of the transformed
        points.
    '''

    if points.size == 0:
        res = np.zeros_like(points)
        return res

    xp = cupy.get_array_module(points, transform)

    row_length = points.shape[1]
    num_points_per_row = row_length // 2

    if num_points_per_row * 2 < row_length:
        raise ValueError(
            f"apply_transform_to_points(): One matrix row has to contain one or more points, and therefore to be divisible by 2. Got a row length of {row_length} instead."
        )

    res = xp.zeros_like(points)

    for i in range(num_points_per_row):
        idx_min = i * 2
        idx_max = idx_min + 2
        points_in = xp.transpose(points[:, idx_min:idx_max])
        points_in_homog = xp.pad(points_in, ((0, 1), (0, 0)), constant_values=1.0)
        points_out = transform @ points_in_homog
        res[:, idx_min:idx_max] = xp.transpose(points_out)

    return res


@do_not_convert
def add_post_transform_to_projection_matrix(proj_mat: np.ndarray, transform: np.ndarray) -> np.ndarray:
    '''Add a post-transform to a projection matrix.

    This can e.g. be used to include 2D transformations of the image to a projection matrix, e.g.
    when performing augmentations on the image.

    Note:
        In case of a decomposed projection matrix (i.e. an intrinsic camera matrix and a extrinsic camera
        pose matrix), the post-transform should be only applied to the intrinsic camera matrix.

    Args:
        proj_mat: A 2D array of shape (3, 4) containing the projection matrix.
        transform: A 2D array of shape (3, 3) containing the affine transformation matrix.
    Returns:
        A 2D array of shape (3, 4) containing the projection matrix with the post-transform applied.
    '''

    xp = cupy.get_array_module(proj_mat, transform)
    to_concat = np.array([[0.0, 0.0, 1.0]], dtype=transform.dtype)
    transform_full = xp.concatenate([transform, to_concat], axis=0)
    res = transform_full @ proj_mat
    return res


@do_not_convert
def apply_clipping_and_get_with_clipping_info(
    rects, centers: np.ndarray, scaling_trafo: np.ndarray, image_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''Apply clipping to a set of rectangles and get the size, area fraction and center of the clipped rectangles.

    Note that a scaling transformation can be defined and will be applied before the clipping. This
    is e.g. useful when the bounding boxes are defined for the original image size, but need to be used
    for a down-sampled image (e.g. a heatmap as used for object detection).

    The scaling transformation is defined as a 3x3 matrix in homogeneous coordinates, and is applied to the
    rectangles before the clipping.

    Important:
        Note that while any affine transformation can be input to this function, the function assumes
        that the input rectangles are defined by their upper left and lower right corners as provided in the input
        and transformed by the scaling transformation. Some transformations (e.g. rotations) may not preserve
        this property, resulting in incorrect rectangles in the output.

    Args:
        rects: A 2D array of shape (N, 4) containing the coordinates of the rectangles.
        centers: A 2D array of shape (N, 2) containing the centers of the rectangles.
        scaling_trafo: A 2D array of shape (3, 3) containing the affine transformation matrix.
        image_hw: A tuple of two integers containing the height and width of the image.
    Returns:
        A tuple containing:
            - A 2D array of shape (N, 4) containing the coordinates of the clipped rectangles.
            - A 2D array of shape (N, 2) containing the height and width of the clipped rectangles.
            - A 2D array of shape (N, 1) containing the area fraction of the clipped rectangles
              (compared to the full, non-clipped area).
            - A 2D array of shape (N, 2) containing the centers of the clipped rectangles.
    '''
    xp = cupy.get_array_module(rects, image_hw)

    def make_homog(points):
        res = xp.pad(points, ((0, 1), (0, 0)), constant_values=1.0)
        return res

    points_1 = make_homog(rects[:, :2].transpose())
    points_2 = make_homog(rects[:, 2:].transpose())

    points_1_transformed = scaling_trafo @ points_1
    points_2_transformed = scaling_trafo @ points_2

    rects_scaled = xp.concatenate([points_1_transformed, points_2_transformed], axis=0).transpose()

    rects_clipped = xp.zeros_like(rects)
    rects_clipped[:, 0] = xp.clip(rects_scaled[:, 0], 0, image_hw[1] - 1)
    rects_clipped[:, 2] = xp.clip(rects_scaled[:, 2], 0, image_hw[1] - 1)
    rects_clipped[:, 1] = xp.clip(rects_scaled[:, 1], 0, image_hw[0] - 1)
    rects_clipped[:, 3] = xp.clip(rects_scaled[:, 3], 0, image_hw[0] - 1)
    h_clipped = xp.abs(rects_clipped[:, 3] - rects_clipped[:, 1])
    w_clipped = xp.abs(rects_clipped[:, 2] - rects_clipped[:, 0])
    h_orig = xp.abs(rects_scaled[:, 3] - rects_scaled[:, 1])
    w_orig = xp.abs(rects_scaled[:, 2] - rects_scaled[:, 0])

    hw_clipped = xp.stack([h_clipped, w_clipped], axis=1)

    area_no_clip = np.multiply(h_orig, w_orig)
    area_clip = np.multiply(h_clipped, w_clipped)

    fraction_area = area_clip / area_no_clip

    centers_homog = make_homog(centers.transpose())
    centers_homog_transformed = scaling_trafo @ centers_homog
    centers_clipped_x = xp.clip(centers_homog_transformed[0, :], 0, image_hw[1] - 1)
    centers_clipped_y = xp.clip(centers_homog_transformed[1, :], 0, image_hw[0] - 1)
    centers_clipped = xp.stack([centers_clipped_x, centers_clipped_y], axis=1)

    return rects_clipped, centers_clipped, hw_clipped, fraction_area


@do_not_convert
def get_is_active(
    hw: np.ndarray,
    classes: Optional[np.ndarray],
    fraction_areas: np.ndarray,
    min_object_size: Optional[np.ndarray],
    per_class_min_object_sizes: Optional[np.ndarray],
    num_classes: int,
    min_fraction_area_thresh: float,
) -> np.ndarray:
    '''Get the indices of the active detections.

    Important:
        This function supports using or not using class labels. If class labels are used,
        per-class size thresholds can be provided. Note that this means that some of the inputs
        can be None (in case they are not used). When used in a DALI
        python operator, ``None`` cannot be directly passed. Therefore, unused inputs need to
        be removed from the function signature. This can be achieved by wrapping this function in a lambda
        function, and expose only the used inputs to the caller, while setting the unused inputs to ``None``
        inside the lambda function.

    Args:
        hw: A 2D array of shape (N, 2) containing the height and width of the detections.
        classes: A 1D array of shape (N,) containing the classes of the detections.
        fraction_areas: A 1D array of shape (N,) containing the area fraction of the detections.
        min_object_size: A 1D array of shape (2,) containing the minimum object size thresholds.
            Must be set to ``None`` if ``per_class_min_object_sizes`` is not ``None``. If both are ``None``,
            no checks for minimum object size are performed.
        per_class_min_object_sizes: A 2D array of shape (num_classes, 2) containing the minimum object size
            thresholds per class. Must be set to ``None`` if ``min_object_size`` is not ``None``. If both
            are ``None``, no checks for minimum object size are performed.
        num_classes: An integer containing the number of classes.
            If ``classes`` is ``None``, this is ignored.
        min_fraction_area_thresh: A float containing the minimum fraction of the area of the detection that
            needs to be present in the image.
    Returns:
        A 1D array of shape (N,) containing the indices of the active detections.
    '''

    if classes is not None:
        xp = cupy.get_array_module(hw, classes, fraction_areas)
        active_classes = classes < num_classes
        classes_for_checking = xp.copy(classes)
        # Set the elements with class IDs >= num_classes (i.e. unconsidered classes) to some class < num_classes.
        # These elements are already not active anyway, but the classes_for_checking array is used as indices in 'per_class_ignore_object_sizes'.
        # As using unconsidered classes would lead to out-of-bounds errors, set to some considered class ID (which 0 always is). As the elements with
        # uncosidered classes are already marked as inactive, the results of the following operations do not matter for these elements.
        classes_for_checking[xp.logical_not(active_classes)] = 0

        if per_class_min_object_sizes is not None:
            active_size_clipped = xp.logical_and(
                hw[:, 0] >= per_class_min_object_sizes[classes_for_checking, 0],
                hw[:, 1] >= per_class_min_object_sizes[classes_for_checking, 1],
            )
        elif min_object_size is not None:
            active_size_clipped = xp.logical_and(
                hw[:, 0] >= min_object_size[0],
                hw[:, 1] >= min_object_size[1],
            )
        else:
            active_size_clipped = xp.ones(hw.shape[0], dtype=xp.bool)
    else:
        xp = cupy.get_array_module(hw, fraction_areas)
        # No category information: accept all classes
        active_classes = xp.ones(hw.shape[0], dtype=xp.bool)
        if min_object_size is not None:
            active_size_clipped = xp.logical_and(
                hw[:, 0] >= min_object_size[0],
                hw[:, 1] >= min_object_size[1],
            )
        else:
            active_size_clipped = xp.ones(hw.shape[0], dtype=xp.bool)

    active_area_fraction = fraction_areas >= min_fraction_area_thresh

    active = xp.logical_and(active_classes, xp.logical_and(active_size_clipped, active_area_fraction))
    active = active.astype(xp.bool)

    return active


@do_not_convert
def pad_to_common_size(*inputs, fill_value: float) -> tuple[np.ndarray, ...]:
    '''Pad a set of inputs to the common size.

    Args:
        inputs: A tuple of input arrays.
        fill_value: A float containing the value to fill the padded areas with.
    Returns:
        A tuple of padded arrays.
    '''

    num_dims = len(inputs[0].shape)
    input_shapes = np.stack([np.array(inp.shape) for inp in inputs], axis=0)
    max_shape = np.max(input_shapes, axis=0, keepdims=False)
    padded = tuple(
        [
            np.pad(
                inp,
                [(0, max_shape[d] - inp.shape[d]) for d in range(num_dims)],
                mode='constant',
                constant_values=fill_value,
            )
            for inp in inputs
        ]
    )
    return padded


if __name__ == "__main__":
    a = np.ones((4, 5))
    b = np.ones((3, 6))
    c = np.ones((7, 1))

    res_a, res_b, res_c = pad_to_common_size(a, b, c, fill_value=0.0)

    print(res_a)
    print(res_b)
    print(res_c)
