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

from typing import Sequence, Union, Tuple, Optional

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np

import nvidia.dali.types as types
import nvidia.dali.fn as fn

from ..pipeline.sample_data_group import SampleDataGroup

from .pipeline_step_base import PipelineStepBase

from ..operators_impl import numba_operators as numba_ops


class BEVBBoxesTransformer3D(PipelineStepBase):
    '''Augment BEV bounding boxes (and related geometry) with rotation, scaling, and translation.

    The augmentation is applied in world coordinates. Related sensor geometry (e.g., extrinsics) is defined
    in ego coordinates and is updated accordingly using provided ego<->world transforms.

    The individual augmentation steps are applied in the following order:
      1. Rotation
      2. Scaling
      3. Translation
    '''

    def __init__(
        self,
        data_field_names_points: Optional[Union[str, int, Sequence[Union[str, int]]]],
        data_field_names_velocities: Optional[Union[str, int, Sequence[Union[str, int]]]],
        data_field_names_sizes: Optional[Union[str, int, Sequence[Union[str, int]]]],
        data_field_names_orientation: Optional[Union[str, int, Sequence[Union[str, int]]]],
        data_field_names_proj_matrices_and_extrinsics: Optional[Union[str, int, Sequence[Union[str, int]]]],
        data_field_names_ego_to_world: Optional[Union[str, int, Sequence[Union[str, int]]]],
        data_field_names_world_to_ego: Optional[Union[str, int, Sequence[Union[str, int]]]],
        rotation_range: Optional[Tuple[float, float]],
        rotation_axis: Optional[int],
        scaling_range: Optional[Tuple[float, float]],
        translation_max_abs: Optional[Tuple[float, float]],
    ):
        '''
        Args:
            data_field_names_points: Name or names of data fields in the input :class:`SampleDataGroup`
                instance containing the points representing the bounding box center (in ``[x, y, z]`` format).
                Optional; will be updated if provided.
            data_field_names_velocities: Name or names of data fields in the input :class:`SampleDataGroup`
                instance containing the velocities of the objects (bounding boxes) (in ``[vx, vy, vz]``
                format).
                Optional; will be updated if provided.
            data_field_names_sizes: Name or names of data fields in the input :class:`SampleDataGroup`
                instance containing the sizes of the bounding boxes (in ``[x, y, z]`` format).
                Optional; will be updated if provided.
            data_field_names_orientation: Name or names of data fields in the input :class:`SampleDataGroup`
                instance containing the orientations of the bounding boxes (in radians).
                Optional; will be updated if provided.
            data_field_names_proj_matrices_and_extrinsics: Name or names of data fields in the input
                :class:`SampleDataGroup` instance containing projection matrices and/or extrinsics. Note that
                camera intrinsics don't need to be adjusted and must not be included in this list. Optional;
                will be updated if provided.
            data_field_names_ego_to_world: Name or names of data fields in the input :class:`SampleDataGroup`
                instance containing matrices representing a transformation (e.g. for points) from ego to
                world coordinates. Optional; will be updated if provided.
            data_field_names_world_to_ego: Name or names of data fields in the input :class:`SampleDataGroup`
                instance containing matrices representing a transformation (e.g. for points) from world to
                ego coordinates. Optional; will be updated if provided.
            rotation_range: Rotation range for the randomized rotation in the augmentation transformation.
                Optional; if not provided, no rotation is applied.
            rotation_axis: Axis of rotation (``0`` indicating ``x``, ``1`` indicating ``y``,
                and ``2`` indicating ``z``). Must be provided if ``rotation_range`` is provided.
            scaling_range: Scaling range for the augmentation transformation.
                Optional; if not provided, no scaling is applied.
            translation_max_abs: Maximum absolute translation range in all dimensions.
                Optional; if not provided, no translation is applied.
        '''

        def int_str_to_list(data):
            if isinstance(data, (str, int)):
                data = [data]
            return data

        self._do_rotate = rotation_range is not None
        self._do_scale = scaling_range is not None
        self._do_translate = translation_max_abs is not None

        if self._do_rotate:
            assert (
                rotation_axis is not None
            ), "If `rotation_range` is set, `rotation_axis` needs to be set too"
            assert (
                len(rotation_range) == 2
            ), "If `rotation_range` is set, it must have 2 elements (minimum and maximum rotation angle)."
            self._rotation_range = [np.float32(rr) for rr in rotation_range]
            self._rotation_axis = rotation_axis
            self._rotation_axis_vec = np.zeros(3, dtype=np.float32)
            self._rotation_axis_vec[rotation_axis] = 1.0

        if self._do_scale:
            assert (
                len(scaling_range) == 2
            ), "If `scaling_range` is set, it must have 2 elements (minimum and maximum scaling factor)."
            self._scaling_range = [np.float32(sr) for sr in scaling_range]

        if self._do_translate:
            assert (
                len(translation_max_abs) == 3
            ), "If `translation_max_abs` is set, it must have 3 elements (one per dimension to translate in)."
            self._translation_max_abs = [np.float32(tma) for tma in translation_max_abs]

        self._data_fields = {}
        if data_field_names_points is not None:
            self._data_fields["data_field_names_points"] = int_str_to_list(data_field_names_points)
        else:
            self._data_fields["data_field_names_points"] = []

        if data_field_names_velocities is not None:
            self._data_fields["data_field_names_velocities"] = int_str_to_list(data_field_names_velocities)
        else:
            self._data_fields["data_field_names_velocities"] = []

        if data_field_names_sizes is not None:
            self._data_fields["data_field_names_sizes"] = int_str_to_list(data_field_names_sizes)
        else:
            self._data_fields["data_field_names_sizes"] = []

        if data_field_names_orientation is not None:
            self._data_fields["data_field_names_orientation"] = int_str_to_list(data_field_names_orientation)
        else:
            self._data_fields["data_field_names_orientation"] = []

        if data_field_names_proj_matrices_and_extrinsics is not None:
            self._data_fields["data_field_names_proj_matrices_and_extrinsics"] = int_str_to_list(
                data_field_names_proj_matrices_and_extrinsics
            )
        else:
            self._data_fields["data_field_names_proj_matrices_and_extrinsics"] = []

        if data_field_names_ego_to_world is not None:
            self._data_fields["data_field_names_ego_to_world"] = int_str_to_list(
                data_field_names_ego_to_world
            )
        else:
            self._data_fields["data_field_names_ego_to_world"] = []

        if data_field_names_world_to_ego is not None:
            self._data_fields["data_field_names_world_to_ego"] = int_str_to_list(
                data_field_names_world_to_ego
            )
        else:
            self._data_fields["data_field_names_world_to_ego"] = []

        assert len(self._data_fields) > 0, "At least one kind of data field name must be set."

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        to_apply_rotation_to = [
            "data_field_names_points",
            "data_field_names_velocities",
            "data_field_names_ego_to_world",
            "data_field_names_world_to_ego",
            "data_field_names_proj_matrices_and_extrinsics",
        ]
        apply_rotation_from_right = [False, False, True, False, True]
        apply_rotation_inv = [False, False, True, False, True]
        apply_rotation_is_transposed = [True, True, False, False, False]
        apply_rotation_make_homog = [True, True, False, False, False]
        # ego' -> ego -> world; note that (ego' -> ego) is the same transformation as for the objects, but it needs to be multiplied from the right
        # to_apply_rotation_to_from_right = ["data_field_names_ego_to_world"]
        # to_apply_rotation_to_inv = ["data_field_names_world_to_ego"]

        orientation_to_apply_rotation_to = "data_field_names_orientation"

        to_apply_scaling_to = [
            "data_field_names_points",
            "data_field_names_velocities",
            "data_field_names_sizes",
            "data_field_names_ego_to_world",
            "data_field_names_world_to_ego",
            "data_field_names_proj_matrices_and_extrinsics",
        ]
        # Note that for scaling, for square inputs it does not matter if the scaling matrix is multiplied from the left or the right. However, for non-square inputs, it matters (and the number of dimensions may not match
        # if done wrongly).
        apply_scaling_from_right = [False, False, False, True, False, True]
        apply_scaling_inv = [False, False, False, True, False, True]
        apply_scaling_transposed = [True, True, True, False, False, False]
        apply_scaling_make_homog = [True, True, True, False, False, False]

        to_apply_translation_to = [
            "data_field_names_points",
            "data_field_names_ego_to_world",
            "data_field_names_world_to_ego",
            "data_field_names_proj_matrices_and_extrinsics",
        ]
        apply_translation_from_right = [False, True, False, True]
        apply_translation_inv = [False, True, False, True]
        apply_translation_transposed = [True, False, False, False]
        apply_translation_make_homog = [True, False, False, False]

        # rotation_angle_deg = self._get_random_in_range(*self._rotation_range)
        # rotation_angle_rad = rotation_angle_deg / 180.0 * np.pi
        if self._do_rotate:
            rotation_angle_rad = self._get_random_in_range(*self._rotation_range)
            rotation_vector = (
                fn.constant(fdata=list(self._rotation_axis_vec), shape=[3], dtype=types.DALIDataType.FLOAT)
                * rotation_angle_rad
            )
            # print(f"/////////////////////////////////////////////////////////////////// rotation_vector: {rotation_vector}")
            # debug_help.print_tensor_op(rotation_vector, "rotation_vector")
            rotation_matrix = numba_ops.get_rot_mat_from_rot_vector(rotation_vector, True)

        if self._do_scale:
            scaling_factor = self._get_random_in_range(*self._scaling_range)
            scaling_matrix = numba_ops.get_scaling_mat_from_vector(
                fn.stack(scaling_factor, scaling_factor, scaling_factor), True
            )

        if self._do_translate:
            translation = [
                self._get_random_in_range(-max_abs_trans, max_abs_trans)
                for max_abs_trans in self._translation_max_abs
            ]
            translation = fn.stack(*translation)

            translation_matrix = numba_ops.get_translation_mat_from_vector(translation)

        if self._do_rotate:
            # Apply rotation transformation to all data fields (except the orientation)
            for key_of_names, from_right, invert, data_transposed, make_homog in zip(
                to_apply_rotation_to,
                apply_rotation_from_right,
                apply_rotation_inv,
                apply_rotation_is_transposed,
                apply_rotation_make_homog,
            ):
                for name in self._data_fields[key_of_names]:
                    paths = data.find_all_occurrences(name)
                    for path in paths:
                        parent = data.get_parent_of_path(path)
                        # Note that instead of `matrix_is_inverted`, invert is passed to the function as `matrix_is_transposed`.
                        # This is as for rotation matrices, the transpose is the inverse and computing the transpose is cheaper.
                        # debug_help.print_tensor_op(parent[name], name + ":pre_rotate")
                        res = numba_ops.apply_matrix(
                            parent[name],
                            rotation_matrix,
                            make_apply_to_homog=make_homog,
                            to_apply_to_is_transposed=data_transposed,
                            matrix_is_transposed=invert,
                            matrix_is_inverted=False,
                            multiply_matrix_from_right=from_right,
                        )
                        parent[name] = res
                        # debug_help.print_tensor_op(parent[name], name + ":post_rotate")

            # Adjust the orientation
            for name in self._data_fields[orientation_to_apply_rotation_to]:
                paths = data.find_all_occurrences(name)
                for path in paths:
                    parent = data.get_parent_of_path(path)
                    orientation = parent[name] + rotation_angle_rad
                    orientation = numba_ops.ensure_range(
                        orientation, -np.pi, np.pi, 2.0 * np.pi, 1, parent.get_type_of_field(name)
                    )
                    parent[name] = orientation

        if self._do_scale:
            for key_of_names, from_right, invert, data_transposed, make_homog in zip(
                to_apply_scaling_to,
                apply_scaling_from_right,
                apply_scaling_inv,
                apply_scaling_transposed,
                apply_scaling_make_homog,
            ):
                for name in self._data_fields[key_of_names]:
                    paths = data.find_all_occurrences(name)
                    for path in paths:
                        parent = data.get_parent_of_path(path)
                        res = numba_ops.apply_matrix(
                            parent[name],
                            scaling_matrix,
                            make_apply_to_homog=make_homog,
                            to_apply_to_is_transposed=data_transposed,
                            matrix_is_transposed=False,
                            matrix_is_inverted=invert,
                            multiply_matrix_from_right=from_right,
                        )
                        parent[name] = res

        if self._do_translate:
            for key_of_names, from_right, invert, data_transposed, make_homog in zip(
                to_apply_translation_to,
                apply_translation_from_right,
                apply_translation_inv,
                apply_translation_transposed,
                apply_translation_make_homog,
            ):
                for name in self._data_fields[key_of_names]:
                    paths = data.find_all_occurrences(name)
                    for path in paths:
                        parent = data.get_parent_of_path(path)
                        res = numba_ops.apply_matrix(
                            parent[name],
                            translation_matrix,
                            make_apply_to_homog=make_homog,
                            to_apply_to_is_transposed=data_transposed,
                            matrix_is_transposed=False,
                            matrix_is_inverted=invert,
                            multiply_matrix_from_right=from_right,
                        )
                        parent[name] = res

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        for field_type in self._data_fields:
            for field_name in self._data_fields[field_type]:
                paths = data_empty.find_all_occurrences(field_name)
                if len(paths) == 0:
                    raise ValueError(f"No fields found with name '{field_name}'")
                for path in paths:
                    parent = data_empty.get_parent_of_path(path)
                    if not parent.is_data_field(field_name):
                        raise ValueError(
                            f"Field with name '{field_name}' in path `{path}` is not a data field."
                        )
        return data_empty

    @staticmethod
    def _get_random_in_range(min, max):
        if min == max:
            res = min
        else:
            min = fn.cast(min, dtype=types.DALIDataType.FLOAT)
            max = fn.cast(max, dtype=types.DALIDataType.FLOAT)
            res = fn.random.uniform(range=fn.stack(min, max))
        return res
