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

from typing import Sequence, Optional

import nvidia.dali.plugin.numba.fn.experimental as fnex
import nvidia.dali.fn as fn
from nvidia.dali import types
from nvidia.dali.pipeline import do_not_convert, DataNode
import numpy as np


@do_not_convert
def pad_to_size(input: DataNode, num_dims: int, data_type: types.DALIDataType, size: int) -> DataNode:
    '''Pad input data to a specified size along the first dimension (i.e. ``dim==0``).

    :cpu:

    Args:
        input: Data to pad.
        num_dims: Number of dimensions in the data.
        data_type: Data type of the input data.
        size: Target size to pad the first dimension (i.e. ``dim==0``) to.

    Returns:
        Padded data with the specified size.
    '''

    def setup_pad_to_size(outs, ins):
        for i in range(len(outs)):
            for sample_idx in range(len(outs[i])):
                outs[i][sample_idx][0] = size
                for j in range(1, len(outs[i][sample_idx].shape)):
                    outs[i][sample_idx][j] = ins[i][sample_idx][i]

    def run_pad_to_size(out0, in0):
        numel_fill = in0.shape[0] if in0.shape[0] < size else size
        for j in range(numel_fill):
            out0[j] = in0[j]
        for j in range(in0.shape[0], size):
            out0[j] = 0.0

    assert num_dims > 0, "num_dims must be greater than 0"

    res = fnex.numba_function(
        input,
        run_fn=run_pad_to_size,
        setup_fn=setup_pad_to_size,
        in_types=[data_type],
        out_types=[data_type],
        outs_ndim=[num_dims],
        ins_ndim=[num_dims],
    )
    return res


@do_not_convert
def remove_inactive(
    data: DataNode, active_mask: DataNode, masked_dimension: int, num_dims: int, data_type: types.DALIDataType
) -> DataNode:
    '''Remove inactive elements in the data.

    :cpu:

    Args:
        data: Data to process.
        active_mask: Boolean mask indicating which elements are active.
        masked_dimension: The dimension of the data along which the elements are located, i.e. where each
            index indicates another element. The mask will be applied in this dimension and its size must
            match the mask size.
        num_dims: The overall number of dimensions in ``data``.
        data_type: Data type of the data.

    Returns:
        Resulting data without inactive elements.

    '''

    def setup_active_zero_padded(outs, ins):
        for sample_idx in range(len(outs[0])):
            for dim in range(len(outs[0][sample_idx])):
                outs[0][sample_idx][dim] = ins[0][sample_idx][dim]
            outs[1][sample_idx][0] = 1

    def get_active_padded_first_dim(out_data, out_num_active, in_data, in_mask):
        active_mask = in_mask != 0
        res_active = in_data[active_mask, ...]
        out_data[: res_active.shape[0], ...] = res_active
        out_num_active[:] = res_active.shape[0]

    assert num_dims > 0, "num_dims must be greater than 0"

    # Make permutation to switch between the first and the masked dimension. Works both for the forward and backward step
    switch_first_dim_permutation = list(range(num_dims))
    switch_first_dim_permutation[0] = masked_dimension
    switch_first_dim_permutation[masked_dimension] = 0

    # @TODO: use BOOL directly?

    data_permutated = fn.transpose(data, perm=switch_first_dim_permutation)
    res_permutated_padded, num_active_elements_as_arr = fnex.numba_function(
        data_permutated,
        fn.cast(active_mask, dtype=types.DALIDataType.UINT8),
        run_fn=get_active_padded_first_dim,
        setup_fn=setup_active_zero_padded,
        in_types=[data_type, types.DALIDataType.UINT8],
        out_types=[data_type, types.DALIDataType.INT32],
        outs_ndim=[num_dims, 1],
        ins_ndim=[num_dims, 1],
    )
    res_permutated = res_permutated_padded[0 : num_active_elements_as_arr[0]]
    res = fn.transpose(res_permutated, perm=switch_first_dim_permutation)

    return res


@do_not_convert
def ensure_range(
    data: DataNode,
    min_value: float,
    max_value: float,
    period: float,
    num_dims: int,
    data_type: types.DALIDataType,
) -> DataNode:
    '''Ensure that the data (containing values from a periodic range) is in a given range (which may be
    larger than one period).

    :cpu:

    Args:
        data: Data to process.
        min_value: Minimum border of the range to bring values into.
        max_value: Maximum border of the range to bring values into.
        period: Period of the data values one or multiple periods are added or subtracted from the values to
            bring them into the desired range.
        num_dims: Number of dimensions in the data. For scalar data, set to ``0``.
        data_type: Data type of ``data``.

    Returns:
        Resulting data where the elements are in the range.
    '''

    def run_ensure_range(out_data, in_data):
        out_data[:] = in_data[:]
        too_low_mask = out_data < min_value
        too_high_mask = out_data > max_value
        if np.any(too_low_mask):
            to_change = out_data[too_low_mask]
            diff = min_value - to_change
            to_add = np.ceil(diff / period) * period
            out_data[too_low_mask] += to_add
        if np.any(too_high_mask):
            to_change = out_data[too_high_mask]
            diff = to_change - max_value
            to_subtract = np.ceil(diff / period) * period
            out_data[too_high_mask] -= to_subtract

    # cannot import 0D arrays to the numba OP; wrap scalars to 1D for the OP and squeeze back afterwards
    data_for_op = data
    op_num_dims = num_dims
    if num_dims == 0:
        data_for_op = fn.expand_dims(data, axes=0)
        op_num_dims = 1

    res = fnex.numba_function(
        data_for_op,
        run_fn=run_ensure_range,
        setup_fn=None,
        in_types=[data_type],
        out_types=[data_type],
        outs_ndim=[op_num_dims],
        ins_ndim=[op_num_dims],
    )

    if num_dims == 0:
        res = fn.squeeze(res, axes=0)
    return res


@do_not_convert
def replace_nans(
    data: DataNode,
    replacement_value: float,
    num_dims: int,
    data_type: types.DALIDataType = types.DALIDataType.FLOAT,
) -> DataNode:
    '''Replace ``nan`` values in data.

    :cpu:

    Args:
        data: Data to apply to.
        replacement_value: Value to replace ``nan`` values with.
        num_dims: Number of dimensions in the data.
        data_type: Data type of ``data``.

    Returns:
        Data with ``nans`` replaced by ``replacement_value``.

    '''

    def run_replace_nans(out_data, in_data):

        out_data[:] = in_data[:]

        flat_data = out_data.reshape(-1)

        for i in range(len(flat_data)):
            if np.isnan(flat_data[i]):
                flat_data[i] = replacement_value

        out_data[:] = flat_data.reshape(out_data.shape)[:]

    assert num_dims > 0, "num_dims must be greater than 0"

    res = fnex.numba_function(
        data,
        run_fn=run_replace_nans,
        setup_fn=None,
        in_types=[data_type],
        out_types=[data_type],
        outs_ndim=[num_dims],
        ins_ndim=[num_dims],
    )
    return res


@do_not_convert
def check_bbox_visibiity(
    bboxes: DataNode, depths: DataNode, image_hw: DataNode, shrink_bbox_to_obtain_int_coords: bool = False
) -> DataNode:
    '''Check visibility of bounding boxes.

    :cpu:

    A bounding box is considered visible if it is not completely occluded by other bounding boxes with
    smaller depth (i.e. fully covered by other boxes).

    Args:
        bboxes: Bounding boxes to process. Data elements are expected to be of type
            :class:`nvidia.dali.types.DALIDataType.FLOAT`. Expected to be a matrix with each row containing
            one bounding box, i.e. first the upper left and then the lower right corner in the format
            ``[x1, y1, x2, y2]``.
        depths: Depths of the individual bounding boxes. Data elements are expected to be of type
            :class:`nvidia.dali.types.DALIDataType.FLOAT`.
        image_hw: The height and width of the image in which the bounding boxes are used. Data elements are
            expected to be of type :class:`nvidia.dali.types.DALIDataType.INT32`.
        shrink_bbox_to_obtain_int_coords: Whether to shrink the bounding box to obtain integer coordinates.
            If ``True``, the bounding box is shrunk to the nearest integer coordinates. If ``False``, the bounding box
            is expanded instead.

    Returns:
        Boolean mask indicating for each bounding box whether it is visible.
    '''

    def setup_result_size(outs, ins):
        for sample_idx in range(len(ins[0])):
            sample_size = ins[0][sample_idx][0]
            outs[0][sample_idx][0] = sample_size

    def perform_check(out_mask, in_bboxes, in_depths, in_image_hw):

        # Note that in_image_hw cannot be passed directly when using Numba, and instead the correct argument type needs to be used
        canvas = np.ones((in_image_hw[0], in_image_hw[1]), dtype=np.int32) * -1

        depth_ordered_indices = np.argsort(-in_depths)

        for doi in depth_ordered_indices:
            box = in_bboxes[doi]
            # The bounding box may have been reflected (e.g. due to scaling with negative scaling factor). Therefore, first extract
            # the minimum and maximum values for both x and y instead of assuming that the first point is the upper left corner.
            if box[0] < box[2]:
                min_x = box[0]
                max_x = box[2]
            else:
                min_x = box[2]
                max_x = box[0]
            if box[1] < box[3]:
                min_y = box[1]
                max_y = box[3]
            else:
                min_y = box[3]
                max_y = box[1]

            # Get the minimum and maximum points as int values
            if shrink_bbox_to_obtain_int_coords:
                min_x = int(np.ceil(min_x))
                min_y = int(np.ceil(min_y))
                max_x = int(np.floor(max_x))
                max_y = int(np.floor(max_y))
            else:
                min_x = int(np.floor(min_x))
                min_y = int(np.floor(min_y))
                max_x = int(np.ceil(max_x))
                max_y = int(np.ceil(max_y))

            # Check if the bounding box is completely outside the image
            if min_x > in_image_hw[1] or max_x < 0 or min_y > in_image_hw[0] or max_y < 0:
                continue

            # Clip the bounding box to the image
            min_x = max(min_x, 0)
            max_x = min(max_x, in_image_hw[1])
            min_y = max(min_y, 0)
            max_y = min(max_y, in_image_hw[0])

            # Draw rect with index of current bounding box
            canvas[min_y:max_y, min_x:max_x] = doi

        # See which indices are still visible (and remove the background of -1)
        unique_sorted = np.unique(canvas)
        if unique_sorted[0] == -1:
            unique_sorted = unique_sorted[1:]

        # Set mask of the visible bboxes to 1 (True)
        out_mask[:] = 0
        out_mask[unique_sorted] = 1

    res = fnex.numba_function(
        bboxes,
        depths,
        image_hw,
        setup_fn=setup_result_size,
        run_fn=perform_check,
        in_types=[types.DALIDataType.FLOAT, types.DALIDataType.FLOAT, types.DALIDataType.INT32],
        out_types=[types.DALIDataType.UINT8],
        ins_ndim=[2, 1, 1],
        outs_ndim=[1],
    )

    res = fn.cast(res, dtype=types.DALIDataType.BOOL)

    return res


@do_not_convert
def check_minimum_bbox_size(bboxes: DataNode, min_size: float, image_hw: DataNode) -> DataNode:
    '''Check whether a bounding box has a minimum size in both dimensions.

    :cpu:

    The bounding box is clipped to the image size before the check is performed.

    Args:
        bboxes: Bounding boxes to process. Data elements are expected to be of type
            :class:`nvidia.dali.types.DALIDataType.FLOAT`. Expected to be a matrix with each row containing
            one bounding box, i.e. first the upper left and then the lower right corner.
        min_size: Minimum size of the bounding box in both dimensions (height, width).
        image_hw: The height and width of the image in which the bounding boxes are used. Data elements are
            expected to be of type :class:`nvidia.dali.types.DALIDataType.INT32`.

    Returns:
        Boolean mask indicating for each bounding box whether it has a minimum size in both dimensions.
    '''

    def setup_result_size(outs, ins):
        for sample_idx in range(len(ins[0])):
            sample_size = ins[0][sample_idx][0]
            outs[0][sample_idx][0] = sample_size

    def perform_check(out_mask, in_bboxes, in_image_hw):
        # Copy bboxes input as it is not allowed to change the inputs and we need to adjust them
        # bboxes_adjusted = np.zeros_like(in_bboxes)
        bboxes_adjusted = np.zeros(in_bboxes.shape, np.float32)
        bboxes_adjusted[:] = in_bboxes[:]
        # Clip x-coords
        bboxes_adjusted[:, 0] = np.clip(bboxes_adjusted[:, 0], 0.0, in_image_hw[1])
        bboxes_adjusted[:, 2] = np.clip(bboxes_adjusted[:, 2], 0.0, in_image_hw[1])
        # Clip y-coords
        bboxes_adjusted[:, 1] = np.clip(bboxes_adjusted[:, 1], 0.0, in_image_hw[0])
        bboxes_adjusted[:, 3] = np.clip(bboxes_adjusted[:, 3], 0.0, in_image_hw[0])
        # GEt bbox size in both dimensions
        diff_x = np.abs(bboxes_adjusted[:, 2] - bboxes_adjusted[:, 0])
        diff_y = np.abs(bboxes_adjusted[:, 3] - bboxes_adjusted[:, 1])
        # Chec whether bbox has at least minimum size in both dimensions
        out_mask[:] = np.logical_and(diff_x >= min_size, diff_y >= min_size)

    res = fnex.numba_function(
        bboxes,
        image_hw,
        setup_fn=setup_result_size,
        run_fn=perform_check,
        in_types=[types.DALIDataType.FLOAT, types.DALIDataType.INT32],
        out_types=[types.DALIDataType.UINT8],
        ins_ndim=[2, 1],
        outs_ndim=[1],
    )

    res = fn.cast(res, dtype=types.DALIDataType.BOOL)

    return res


@do_not_convert
def check_points_in_box(points: DataNode, min_point: Sequence[float], max_point: Sequence[float]) -> DataNode:
    '''Check whether points are inside an axis-aligned box.

    :cpu:

    Args:
        points: Points to check. Each row corresponds to one point. Data elements are expected to be of type
            :class:`nvidia.dali.types.DALIDataType.FLOAT`.
        min_point: Minimum corner (in all dimensions) of the box.
        max_point: Maximum corner (in all dimensions) of the box.

    Returns:
        Boolean mask indicating which points are inside the box.

    '''

    # Will be accessed in perform_check. Note that the convertion to numpy arrays needs to happen here as otherwise it leads to errors.
    # Performing the conversion here is also more efficient as it will happen once at grapgh construction time and treated inside the
    # perform_check as constants.
    min_point_np = np.expand_dims(np.array(min_point), axis=0)
    max_point_np = np.expand_dims(np.array(max_point), axis=0)

    def setup_result_size(outs, ins):
        for sample_idx in range(len(ins[0])):
            sample_size = ins[0][sample_idx][0]
            outs[0][sample_idx][0] = sample_size

    def perform_check(out_mask, in_points):
        element_wise_above_min = in_points >= min_point_np
        element_wise_below_max = in_points <= max_point_np
        element_wise_inside = np.logical_and(element_wise_above_min, element_wise_below_max)

        # workaround due to `axis` argument for `np.min` not working with numba
        for i in range(element_wise_inside.shape[0]):
            out_mask[i] = np.min(element_wise_inside[i, :])

    res = fnex.numba_function(
        points,
        setup_fn=setup_result_size,
        run_fn=perform_check,
        in_types=[types.DALIDataType.FLOAT],
        out_types=[types.DALIDataType.UINT8],
        ins_ndim=[2],
        outs_ndim=[1],
    )

    res = fn.cast(res, dtype=types.DALIDataType.BOOL)

    return res


@do_not_convert
def crop_coordinates(
    points: DataNode, min_point_np: Sequence[float], max_point_np: Sequence[float], dtype: types.DALIDataType
) -> DataNode:
    '''Crop coordinates to a given range.

    :cpu:

    Args:
        points: Points to check. Each row corresponds to one point. Data type is expected to be
            :class:`nvidia.dali.types.DALIDataType.FLOAT`.
            Note that the point may have any number of dimensions (i.e. any number of columns in the input data).
            This can e.g. be used to express bounding boxes as points, where two diagonally opposite points
            are given for each entry, and the two points are combined as ``[min_x, min_y, max_x, max_y]``.
        min_point: Minimum corner (in all dimensions) of the box.
        max_point: Maximum corner (in all dimensions) of the box.

    Returns:
        Points with coordinates cropped to the given range.
    '''

    def setup_result_size(outs, ins):
        for sample_idx in range(len(ins[0])):
            # Copy input shape to output shape properly
            input_shape = ins[0][sample_idx]
            for dim_idx in range(len(input_shape)):
                outs[0][sample_idx][dim_idx] = input_shape[dim_idx]

    def perform_cropping(out_points, in_points):
        for i in range(in_points.shape[0]):
            for j in range(in_points.shape[1]):
                val = in_points[i, j]
                min_val = min_point_np[j]
                max_val = max_point_np[j]
                if val < min_val:
                    out_points[i, j] = min_val
                elif val > max_val:
                    out_points[i, j] = max_val
                else:
                    out_points[i, j] = val

    res = fnex.numba_function(
        points,
        setup_fn=setup_result_size,
        run_fn=perform_cropping,
        in_types=[dtype],
        out_types=[dtype],
        ins_ndim=[2],
        outs_ndim=[2],
    )

    return res


@do_not_convert
def get_rot_mat_from_rot_vector(rot_vector: DataNode, as_homog: bool = False, eps: float = 1e-7) -> DataNode:
    '''Get a rotation matrix from a Rodrigues rotation vector.

    :cpu:

    Args:
        rot_vector: Rodrigues rotation vector. Data elements are expected to be of type
            :class:`nvidia.dali.types.DALIDataType.FLOAT`.
        as_homog: Whether to return a homogeneous transformation matrix.
        eps: Small value to avoid division by zero. Vectors with length below
            this value are considered to be zero.

    Returns:
        Rotation matrix. If ``as_homog`` is ``True``, the matrix is a homogeneous transformation matrix.
    '''

    # Compted at graph construction time and can be used as a constant inside get_matrix()
    identity = np.eye(3, dtype=np.float32)

    def setup_result_size(outs, ins):
        size = 4 if as_homog else 3
        for sample_idx in range(len(ins[0])):
            outs[0][sample_idx][0] = size
            outs[0][sample_idx][1] = size

    def get_matrix(out_matrix, in_vector):

        angle = np.sqrt(
            in_vector[0] * in_vector[0] + in_vector[1] * in_vector[1] + in_vector[2] * in_vector[2]
        )

        if angle < eps:
            rot_mat = identity
        else:
            axis = in_vector / angle

            cross_mat = [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
            cross_mat = np.array(cross_mat, dtype=np.float32)

            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            rot_mat = (identity + sin_angle * cross_mat + (1.0 - cos_angle) * (cross_mat @ cross_mat)).astype(
                np.float32
            )

        if not as_homog:
            out_matrix[:] = rot_mat[:]
        else:
            out_matrix[0:3, 0:3] = rot_mat
            out_matrix[0:3, 3] = 0.0
            out_matrix[3, 0:3] = 0.0
            out_matrix[3, 3] = 1.0

    res = fnex.numba_function(
        rot_vector,
        setup_fn=setup_result_size,
        run_fn=get_matrix,
        in_types=[types.DALIDataType.FLOAT],
        out_types=[types.DALIDataType.FLOAT],
        ins_ndim=[1],
        outs_ndim=[2],
    )

    return res


@do_not_convert
def get_translation_mat_from_vector(translation: DataNode) -> DataNode:
    '''Get a translation matrix from a translation vector.

    :cpu:

    Args:
        translation: Translation vector. Data elements are expected to be of type
            :class:`nvidia.dali.types.DALIDataType.FLOAT`.

    Returns:
        Translation matrix (homogeneous transformation matrix).
    '''

    def setup_result_size(outs, ins):
        size = 4
        for sample_idx in range(len(ins[0])):
            outs[0][sample_idx][0] = size
            outs[0][sample_idx][1] = size

    def get_matrix(out_matrix, in_vector):
        res = np.eye(4, dtype=np.float32)
        res[0:3, 3] = in_vector
        out_matrix[:] = res[:]

    res = fnex.numba_function(
        translation,
        setup_fn=setup_result_size,
        run_fn=get_matrix,
        in_types=[types.DALIDataType.FLOAT],
        out_types=[types.DALIDataType.FLOAT],
        ins_ndim=[1],
        outs_ndim=[2],
    )

    return res


@do_not_convert
def get_scaling_mat_from_vector(scaling: DataNode, as_homog: bool = False) -> DataNode:
    '''Get a scaling matrix from a scaling vector.

    :cpu:

    Args:
        scaling: Vector containing the scaling factors for each dimension. Data elements are expected to be
            of type :class:`nvidia.dali.types.DALIDataType.FLOAT`.
        as_homog: Whether to return a homogeneous transformation matrix.

    Returns:
        Scaling matrix. If ``as_homog`` is ``True``, the matrix is a homogeneous transformation matrix.
    '''

    def setup_result_size(outs, ins):
        size = 4 if as_homog else 3
        for sample_idx in range(len(ins[0])):
            outs[0][sample_idx][0] = size
            outs[0][sample_idx][1] = size

    def get_matrix(out_matrix, in_vector):
        res = np.eye(3, dtype=np.float32)
        res[0, 0] = in_vector[0]
        res[1, 1] = in_vector[1]
        res[2, 2] = in_vector[2]

        if not as_homog:
            out_matrix[:] = res[:]
        else:
            out_matrix[0:3, 0:3] = res
            out_matrix[0:3, 3] = 0.0
            out_matrix[3, 0:3] = 0.0
            out_matrix[3, 3] = 1.0

    res = fnex.numba_function(
        scaling,
        setup_fn=setup_result_size,
        run_fn=get_matrix,
        in_types=[types.DALIDataType.FLOAT],
        out_types=[types.DALIDataType.FLOAT],
        ins_ndim=[1],
        outs_ndim=[2],
    )

    return res


# Inner function as needed in `apply_matrix()`. This is because `@do_not_convert` cannot be used for `apply_matrix()`
# due to if-conditions based on DataNode data (i.e. conditions need to be evaluated at graph execution time,
# not construction time).
@do_not_convert
def _apply_matrix_2d_input_ensured(
    to_apply_to: DataNode,
    matrix: DataNode,
    in_homog: bool,
    to_apply_to_is_transposed: bool,
    matrix_is_transposed: bool,
    matrix_is_inverted: bool,
    multiply_matrix_from_right: bool,
) -> DataNode:

    def setup_result_size(outs, ins):
        for sample_idx in range(len(ins[0])):
            outs[0][sample_idx][:] = ins[0][sample_idx][:]

    def do_apply_matrix(out_applied, in_to_apply_to, in_matrix):
        data = np.zeros(in_to_apply_to.shape, np.float32)
        data[:] = in_to_apply_to[:]

        if to_apply_to_is_transposed:
            data = data.transpose()

        if in_homog:
            ones_to_add = np.ones((1, data.shape[1]), dtype=np.float32)
            data = np.concatenate((data, ones_to_add), axis=0)

        if matrix_is_transposed:
            matrix_to_use = np.transpose(in_matrix)
        else:
            matrix_to_use = in_matrix

        if matrix_is_inverted:
            temp_mat = np.zeros(matrix_to_use.shape, dtype=np.float32)
            temp_mat[:] = matrix_to_use[:]
            matrix_to_use = np.linalg.inv(temp_mat)

        if not multiply_matrix_from_right:
            data = matrix_to_use @ data
        else:
            data = data @ matrix_to_use

        if in_homog:
            data = data[0:-1, :] / data[-1, :]

        if to_apply_to_is_transposed:
            data = data.transpose()

        out_applied[:] = data[:]

    res = fnex.numba_function(
        to_apply_to,
        matrix,
        setup_fn=setup_result_size,
        run_fn=do_apply_matrix,
        in_types=[types.DALIDataType.FLOAT, types.DALIDataType.FLOAT],
        out_types=[types.DALIDataType.FLOAT],
        ins_ndim=[2, 2],
        outs_ndim=[2],
    )
    return res


# Note that this function uses `_apply_matrix_2d_input_ensured()` as the actual operator implementation. This is to avoid using `@do_not_convert`
# on this function, thereby anabling the use of (pipeline run time) conditionals inside it.
def apply_matrix(
    to_apply_to: DataNode,
    matrix: DataNode,
    make_apply_to_homog: bool,
    to_apply_to_is_transposed: bool,
    matrix_is_transposed: bool,
    matrix_is_inverted: bool,
    multiply_matrix_from_right: bool,
) -> DataNode:
    '''Apply a matrix to data (vector or matrix).

    :cpu:

    Args:
        to_apply_to: Data node to apply the matrix to. Data elements are expected to be of type
            :class:`nvidia.dali.types.DALIDataType.FLOAT`.
        matrix: Matrix to apply. Data elements are expected to be of type
            :class:`nvidia.dali.types.DALIDataType.FLOAT`.
        make_apply_to_homog: Whether to make the input homogeneous before applying the matrix.
            The result will be converted back to Euclidean space after applying the matrix.
        to_apply_to_is_transposed: Whether the input is transposed.
            If set, the input will be transposed before applying the matrix, and transposed back afterwards.
        matrix_is_transposed: Whether the matrix is transposed.
            If set, the matrix will be transposed before applying it.
        matrix_is_inverted: Whether the matrix is inverted.
            If set, the matrix will be inverted before applying it.
        multiply_matrix_from_right: Whether to multiply the matrix from the right.
            If set, the matrix will be multiplied from the right (otherwise from the left).

    Returns:
        Data with the matrix applied.
    '''
    # If the input is 1D, the shape of the shape is 1
    is_to_apply_to_1d = to_apply_to.shape().shape()[0] == 1

    if is_to_apply_to_1d:
        to_apply_to = fn.expand_dims(to_apply_to, axes=1)

    res = _apply_matrix_2d_input_ensured(
        to_apply_to,
        matrix,
        make_apply_to_homog,
        to_apply_to_is_transposed,
        matrix_is_transposed,
        matrix_is_inverted,
        multiply_matrix_from_right,
    )

    if is_to_apply_to_1d:
        res = fn.squeeze(res, axes=1)

    return res


@do_not_convert
def get_center_from_bboxes(
    bboxes: DataNode, dtype_in: types.DALIDataType = types.DALIDataType.FLOAT
) -> DataNode:
    '''Get the center of bounding boxes.

    :cpu:

    Args:
        bboxes: Bounding boxes. The first dimension (``dim==0``) iterates over the bounding boxes.
            Each bounding box is expected to be in the format ``[x1, y1, x2, y2]``, where ``(x1, y1)`` and
            ``(x2, y2)`` are the coordinates of the two diagonally opposite corners of the bounding box.
        dtype_in: Data type of the bounding boxes.

    Returns:
        The center of the bounding boxes. The data type of the output is
        :class:`nvidia.dali.types.DALIDataType.FLOAT`.
    '''

    def setup_result_size(outs, ins):
        for sample_idx in range(len(ins[0])):
            outs[0][sample_idx][0] = ins[0][sample_idx][0]
            outs[0][sample_idx][1] = 2

    def get_center(out_center, in_bboxes):
        in_bboxes = in_bboxes.astype(np.float32)
        for i in range(in_bboxes.shape[0]):
            out_center[i, 0] = (in_bboxes[i, 0] + in_bboxes[i, 2]) / np.float32(2.0)
            out_center[i, 1] = (in_bboxes[i, 1] + in_bboxes[i, 3]) / np.float32(2.0)

    res = fnex.numba_function(
        bboxes,
        setup_fn=setup_result_size,
        run_fn=get_center,
        in_types=[dtype_in],
        out_types=[types.DALIDataType.FLOAT],
        ins_ndim=[2],
        outs_ndim=[2],
    )
    return res


@do_not_convert
def get_radii_from_bboxes(
    bboxes: DataNode,
    dtype_bboxes: types.DALIDataType = types.DALIDataType.FLOAT,
    scaling_factor: float = 0.8,
    centers: Optional[DataNode] = None,
    dtype_center: types.DALIDataType = types.DALIDataType.FLOAT,
) -> DataNode:
    '''Compute radii from bounding boxes and a scaling factor.

    :cpu:

    The radius is the minimum distance from the bounding box center to its border scaled by the scaling
    factor. The center to use can be provided as an input (may not correspond to the geometrical center of
    the bounding box), or computed from the bounding boxes.

    Note:
        The bounding boxes are expected to be in the format ``[x1, y1, x2, y2]``, where ``(x1, y1)`` and
        ``(x2, y2)`` are the coordinates of the two diagonally opposite corners of the bounding box. No
        assumption is made on the order of the points or on which diagonal the points are on.

    Args:
        bboxes: Bounding boxes.
        dtype_bboxes: Data type of the bounding boxes.
        scaling_factor: Scaling factor for the radii.
        centers: Centers of the bounding boxes.
        dtype_center: Data type of the center.

    Returns:
        Radii. The data type of the output is :class:`nvidia.dali.types.DALIDataType.FLOAT`.
    '''

    def setup_result_size(outs, ins):
        for sample_idx in range(len(ins[0])):
            outs[0][sample_idx][0] = ins[0][sample_idx][0]

    def get_radii(out_radii, in_bboxes, in_center):
        in_bboxes = in_bboxes.astype(np.float32)
        in_center = in_center.astype(np.float32)
        dists = np.zeros((4,), dtype=np.float32)
        # Note that the bboxes are in the format [x1, y1, x2, y2]
        # and the two points are on the opposite corners of a diagonal of the box.
        # However, they could be on either of the diagonals, and in any order.
        # Therefore, we need to find the top, left, bottom, and right coordinates of the box.
        for i in range(in_bboxes.shape[0]):
            in_center_i = in_center[i, :]
            in_bbox_i = in_bboxes[i, :]

            if in_bbox_i[0] <= in_bbox_i[2]:
                bbox_l = in_bbox_i[0]
                bbox_r = in_bbox_i[2]
            else:
                bbox_l = in_bbox_i[2]
                bbox_r = in_bbox_i[0]
            if in_bbox_i[1] <= in_bbox_i[3]:
                bbox_t = in_bbox_i[1]
                bbox_b = in_bbox_i[3]
            else:
                bbox_t = in_bbox_i[3]
                bbox_b = in_bbox_i[1]

            # Distance to left
            dists[0] = in_center_i[0] - bbox_l
            # Distance to top
            dists[1] = in_center_i[1] - bbox_t
            # Distance to right
            dists[2] = bbox_r - in_center_i[0]
            # Distance to right
            dists[3] = bbox_b - in_center_i[1]

            # The distance is the minimum of the distances to the edges of the bounding box.
            # If the distance is negative, it means that the center is outside the bounding box, and the
            # radius is 0.
            dist = np.maximum(0, np.min(dists))
            radius = dist * scaling_factor
            out_radii[i] = radius

    if centers is None:
        centers = get_center_from_bboxes(bboxes, dtype_in=dtype_bboxes)
        dtype_center = types.DALIDataType.FLOAT

    res = fnex.numba_function(
        bboxes,
        centers,
        setup_fn=setup_result_size,
        run_fn=get_radii,
        in_types=[dtype_bboxes, dtype_center],
        out_types=[types.DALIDataType.FLOAT],
        ins_ndim=[2, 2],
        outs_ndim=[1],
    )
    return res
