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

# Used to enable type hints using a class type inside the implementation of that class itself.
from __future__ import annotations

import nvidia.dali.fn as fn
import nvidia.dali.types as types

from typing import Tuple, Union, Sequence, Optional

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..pipeline.sample_data_group import SampleDataGroup

from ..operators_impl.python_operator_functions import *
from ..operators_impl import numba_operators as numba_ops
from ..internal_helpers.debug_helpers import *

from .pipeline_step_base import PipelineStepBase


class VisibleBboxSelector(PipelineStepBase):
    '''Select visible 2D bounding boxes.

    A box is considered visible if it is not completely overlapped by nearer boxes (occlusion test)
    and/or if it meets a minimum size threshold. The result is written as a boolean mask to the
    configured output path. Both checks are optional and can be enabled or disabled independently.

    A mask is added which indicates which boxes are visible. The original bounding boxes are not modified.

    See also:
        - :class:`AnnotationElementConditionEval` can be used to combine the results of this step with other
          conditions.
        - :class:`ConditionalElementRemoval` can be used to remove elements from the data based on this
          condition or a combination of this condition with other conditions.
    '''

    def __init__(
        self,
        bboxes_field_name: Union[str, int],
        resulting_mask_field_path: Union[str, int, Tuple[Tuple[str, int], ...]],
        image_field_name: Optional[Union[str, int]] = None,
        image_hw_field_name: Optional[Union[str, int]] = None,
        image_hw: Optional[Sequence[int]] = None,
        check_for_bbox_occlusion: bool = True,
        check_for_minimum_size: bool = True,
        depths_field_name: Optional[Union[str, int]] = None,
        minimum_bbox_size: Optional[float] = None,
    ):
        '''

        Note that the step expects exactly one data field in the input :class:`SampleDataGroup` instance to
        contain the bounding boxes (as well as only one field containing the depths). If multiple sets of
        bounding boxes are present in the data, this processing steps has to be applied to parts (sub-trees)
        of the input data individually so that each part contains only one set of bounding boxes, and
        access modifier wrapper steps need to be used (see class :class:`GroupToApplyToSelectedStepBase` and
        its subclasses).

        Args:
            bboxes_field_name: Name of data field in the input :class:`SampleDataGroup` instance containing
                the bounding boxes. Each row is expected to contain a bounding box in the format:
                ``[min_x, min_y, max_x, max_y]``. The input data must contain exactly one field with this
                name.
            resulting_mask_field_path: Path of the data field to store the result as. The path is relative
                to the root element. Note that if this step is wrapped by a sub-tree selection step, the
                root of the selected sub-tree acts as the root.
            image_field_name: Name of field containing the image from which to extract the size. Only one of
                ``image_field_name``, ``image_hw_field_name``, or ``image_hw`` should be set (single source
                of truth).
            image_hw_field_name: Name of field containing the image size for which the bounding boxes are
                defined. Only one of ``image_field_name``, ``image_hw_field_name``, or ``image_hw`` should
                be set (single source of truth).
            image_hw: Image size ``[height, width]`` for the image for which the bounding boxes are defined.
                Only one of ``image_field_name``, ``image_hw_field_name``, or ``image_hw`` should be set
                (single source of truth).
            check_for_bbox_occlusion: Whether to consider boxes invisible if completely overlapped by nearer
                boxes.
            check_for_minimum_size: Whether to consider boxes invisible if below a minimum size.
            depths_field_name: Name of the data field containing the bounding box depth. Needs to be set if
                ``check_for_bbox_occlusion`` is set to ``True``. The input data must contain exactly one field
                with this name.
            minimum_bbox_size: Minimum size of a bounding box to be visible. Needs to be set if
                ``check_for_minimum_size`` is set to ``True``.
        '''

        # Ensure exactly one of image_field_name, image_hw_field_name, or image_hw is set
        num_set = sum([image_field_name is not None, image_hw_field_name is not None, image_hw is not None])
        assert (
            num_set == 1
        ), "Exactly one of 'image_field_name', 'image_hw_field_name', or 'image_hw' must be set (single source of truth)"
        assert (
            check_for_bbox_occlusion or check_for_minimum_size
        ), "At least one of the parameters `check_for_bbox_occlusion` and `check_for_minimum_size` has to be `True`"
        assert (
            not check_for_minimum_size or minimum_bbox_size is not None
        ), "If ``check_for_minimum_size==True``, the parameter ``minimum_size`` has to be set to a non-None value"
        assert (
            not check_for_bbox_occlusion or depths_field_name is not None
        ), "If ``check_for_bbox_occlusion==True``, the parameter ``depths_field_name`` has to be set to a non-None value"

        self._bboxes_field_name = bboxes_field_name
        self._depths_field_name = depths_field_name
        self._image_field_name = image_field_name
        self._image_hw_field_name = image_hw_field_name
        self._image_hw = image_hw
        self._extract_size_from_image = image_field_name is not None
        self._use_fixed_image_hw = image_hw is not None
        self._resulting_mask_field_path = resulting_mask_field_path
        self._check_for_bbox_occlusion = check_for_bbox_occlusion
        self._check_for_minimum_size = check_for_minimum_size
        self._minimum_bbox_size = minimum_bbox_size

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        if self._use_fixed_image_hw:
            image_hw = fn.constant(idata=self._image_hw, dtype=types.DALIDataType.INT32)
        elif self._extract_size_from_image:
            # Extract size from image using .shape() method
            image_paths = data.find_all_occurrences(self._image_field_name)
            image = data.get_item_in_path(image_paths[0])
            image_shape = image.shape()
            # Use fn.stack to create a proper tensor with [height, width]
            # Cast to int32 for consistency with image_hw fields
            image_hw = fn.cast(fn.stack(image_shape[-3], image_shape[-2]), dtype=types.DALIDataType.INT32)
        else:
            # Use size field
            image_hw_paths = data.find_all_occurrences(self._image_hw_field_name)
            image_hw = data.get_item_in_path(image_hw_paths[0])

        bboxes_paths = data.find_all_occurrences(self._bboxes_field_name)
        bboxes = data.get_item_in_path(bboxes_paths[0])

        depths_paths = data.find_all_occurrences(self._depths_field_name)
        depths = data.get_item_in_path(depths_paths[0])

        if self._check_for_bbox_occlusion:
            mask = numba_ops.check_bbox_visibiity(
                bboxes, depths, image_hw, shrink_bbox_to_obtain_int_coords=False
            )
            if self._check_for_minimum_size:
                mask_size = numba_ops.check_minimum_bbox_size(bboxes, self._minimum_bbox_size, image_hw)
                mask = mask & mask_size
        else:  # self._check_for_minimum_size has to be True, as one of the checks needs to be set to True (and this is ensured in the constructor)
            mask = numba_ops.check_minimum_bbox_size(bboxes, self._minimum_bbox_size, image_hw)

        self._add_result_field(data)

        data.set_item_in_path(self._resulting_mask_field_path, mask)

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        num_bboxes_fields = data_empty.get_num_occurrences(self._bboxes_field_name)
        if num_bboxes_fields == 0:
            raise ValueError(f"No bounding box fields found with name: '{self._bboxes_field_name}'.")
        if num_bboxes_fields > 1:
            raise ValueError(
                f"More than one occurence of box fields found with name: '{self._bboxes_field_name}'. Field needs to have an unique name."
            )

        if self._extract_size_from_image:
            # Check for image field
            num_image_fields = data_empty.get_num_occurrences(self._image_field_name)
            if num_image_fields == 0:
                raise ValueError(f"No image field found with name: '{self._image_field_name}'.")
            if num_image_fields > 1:
                raise ValueError(
                    f"More than one occurence of image field found with name: '{self._image_field_name}'. Field needs to have an unique name."
                )
        elif not self._use_fixed_image_hw:
            # Check for size field
            num_image_hw_fields = data_empty.get_num_occurrences(self._image_hw_field_name)
            if num_image_hw_fields == 0:
                raise ValueError(f"No image size field found with name: '{self._image_hw_field_name}'.")
            if num_image_hw_fields > 1:
                raise ValueError(
                    f"More than one occurence of image size field found with name: '{self._image_hw_field_name}'. Field needs to have an unique name."
                )

        if self._check_for_bbox_occlusion:
            num_depths_fields = data_empty.get_num_occurrences(self._depths_field_name)
            if num_depths_fields == 0:
                raise ValueError(f"No depths field found with name: '{self._depths_field_name}'.")
            if num_depths_fields > 1:
                raise ValueError(
                    f"More than one occurence of depths field found with name: '{self._depths_field_name}'. Field needs to have an unique name."
                )

        self._add_result_field(data_empty)

        return data_empty

    def _add_result_field(self, data_inout: SampleDataGroup):
        if data_inout.path_is_single_name(self._resulting_mask_field_path):
            field_name = self._resulting_mask_field_path
            if data_inout.has_child(field_name):
                raise ValueError(
                    f"SampleDataGroup object already has a child with name '{field_name}'. Cannot create the field to store results."
                )
            data_inout.add_data_field(field_name, type=types.DALIDataType.BOOL)
        else:
            parent = data_inout.get_parent_of_path(self._resulting_mask_field_path)
            field_name = self._resulting_mask_field_path[-1]
            if parent.has_child(field_name):
                raise ValueError(
                    f"Group data field at path {self._resulting_mask_field_path[:-1]} already has a child with name '{field_name}'. Cannot create the field to store results."
                )
            parent.add_data_field(field_name, type=types.DALIDataType.BOOL)
