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

from typing import Union, Tuple, Sequence, List

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..operators_impl import numba_operators as numba_op

from ..pipeline.sample_data_group import SampleDataGroup

from .pipeline_step_base import PipelineStepBase


class ConditionalElementRemover(PipelineStepBase):
    '''Remove elements from arrays (e.g., per‑object data) based on a boolean mask.

    Arrays are stored as (multi-dimensional) tensors; for each array a dimension index indicates the element
    axis (the axis along which the elements to be removed/retained are enumerated). Elements with mask
    value ``False`` are removed along the configured dimension for each target field.

    See also:
        Multiple classes are available which evaluate conditions of some kind and store the results as
        boolean masks. These masks can be used in this class:

          - :class:`AnnotationElementConditionEval`
          - :class:`VisibleBboxSelector`
          - :class:`PointsInRangeCheck`
          - :class:`BoundingBoxToHeatmapConverter`
    '''

    def __init__(
        self,
        annotation_field_name: Union[str, int],
        mask_field_name: Union[str, int],
        field_names_to_process: Sequence[Union[str, int]],
        field_dims_to_process: Sequence[int],
        fields_to_process_num_dims: Sequence[int],
        remove_mask_field: bool,
    ):
        '''
        Args:
            annotation_field_name: Name of the annotation data group field to process. Each annotation field
                is processed independently.
            mask_field_name: Name of the boolean mask indicating which elements to keep (``True``) or
                remove (``False``). Must be a child of each annotation field.
            field_names_to_process: Names of fields to process. The fields must be present in each annotation
                field.
            field_dims_to_process: For each field name, the dimension index along which elements are to be
                removed.
            fields_to_process_num_dims: For each field name, the number of dimensions in the tensor.
            remove_mask_field: Whether to remove the mask field after applying this step.

        '''

        assert len(field_names_to_process) == len(
            field_dims_to_process
        ), "Number of elements in `field_names_to_process` and `field_dims_to_process` do not match."
        assert len(field_names_to_process) == len(
            fields_to_process_num_dims
        ), "Number of elements in `field_names_to_process` and `fields_to_process_num_dims` do not match."

        self._annotation_field_name = annotation_field_name
        self._mask_field_name = mask_field_name
        self._field_names_to_process = field_names_to_process
        self._field_dims_to_process = field_dims_to_process
        self._fields_to_process_num_dims = fields_to_process_num_dims
        self._do_remove_mask_field = remove_mask_field

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # Make sure annotations have all needed fields and set output fields
        annotation_paths = data.find_all_occurrences(self._annotation_field_name)
        for ap in annotation_paths:
            annotations = data.get_item_in_path(ap)
            is_active = annotations[self._mask_field_name]
            for name, dim, num_dims in zip(
                self._field_names_to_process, self._field_dims_to_process, self._fields_to_process_num_dims
            ):
                curr_data = annotations[name]
                curr_data_type = annotations.get_type_of_field(name)
                curr_res = numba_op.remove_inactive(curr_data, is_active, dim, num_dims, curr_data_type)
                annotations[name] = curr_res

        if self._do_remove_mask_field:
            self._remove_mask(data)

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        annotation_paths = data_empty.find_all_occurrences(self._annotation_field_name)
        if len(annotation_paths) == 0:
            raise annotation_paths(
                f"No occurrences of annotations found. Annotation data group fields are expected to have the name '{self._annotation_field_name}', as specified in the constructor."
            )
        for ap in annotation_paths:
            annotation = data_empty.get_item_in_path(ap)
            for field in self._field_names_to_process:
                if not field in annotation.contained_top_level_field_names:
                    raise KeyError(f"No field to process '{field}' in annotation at path '{ap}'")
            if not self._mask_field_name in annotation.contained_top_level_field_names:
                raise KeyError(f"No mask field '{self._mask_field_name}' in annotation at path `{ap}`")

        if self._do_remove_mask_field:
            self._remove_mask(data_empty)

        return data_empty

    def _remove_mask(self, data_inout: SampleDataGroup):
        '''Remove the mask field from the annotation data group.

        Args:
            data_inout: Data to be processed by the step.
        '''
        annotation_paths = data_inout.find_all_occurrences(self._annotation_field_name)
        for ap in annotation_paths:
            annotation = data_inout.get_item_in_path(ap)
            annotation.remove_field(self._mask_field_name)
