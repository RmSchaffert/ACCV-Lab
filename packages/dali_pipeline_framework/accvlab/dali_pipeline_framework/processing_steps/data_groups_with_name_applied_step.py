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

from typing import Union, Tuple, Sequence

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..pipeline.sample_data_group import SampleDataGroup

from .group_to_apply_to_selected_step_base import GroupToApplyToSelectedStepBase
from .pipeline_step_base import PipelineStepBase


class DataGroupsWithNameAppliedStep(GroupToApplyToSelectedStepBase):
    '''Apply a contained processing step to all sub-trees whose root is a data group field with a given name.

    The name is defined at construction; all matching data group fields are located and the contained step is
    applied to each corresponding sub-tree.
    '''

    def __init__(
        self,
        processing_step_to_apply: PipelineStepBase,
        names_of_groups_to_apply_to: Union[str, int, Sequence[str, int]],
        check_minimum_one_name_match: bool = True,
    ):
        '''

        Args:
            processing_step_to_apply: Contained processing step to apply.
            names_of_groups_to_apply_to: Name or list of names of data group fields to select as sub-tree roots.
            check_minimum_one_name_match: If ``True``, require that at least one field is found for each
                provided name, and an error is raised otherwise when checking the input.

        '''
        GroupToApplyToSelectedStepBase.__init__(self, processing_step_to_apply)
        if isinstance(names_of_groups_to_apply_to, (str, int)):
            names_of_groups_to_apply_to = [names_of_groups_to_apply_to]
        self._names_of_groups_to_apply_to = names_of_groups_to_apply_to
        self._check_minimum_one_name_match = check_minimum_one_name_match

    @override
    def _check_and_get_paths_to_apply_to(self, data: SampleDataGroup) -> Sequence[Sequence[str, int]]:

        paths = []
        for name in self._names_of_groups_to_apply_to:
            paths_field_name = data.find_all_occurrences(name)
            if self._check_minimum_one_name_match and len(paths_field_name) == 0:
                raise ValueError(f"DataGroupsWithNameAppliedStep: No fields with name `{name}` found.")
            for path in paths_field_name:
                if not data.path_exists_and_is_data_group_field(path):
                    raise ValueError(
                        f"DataGroupsWithNameAppliedStep: Field in path `{path}` is not a data group field."
                    )
            paths = paths + paths_field_name
        return paths
