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

from typing import Union, Sequence

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ..pipeline.sample_data_group import SampleDataGroup

from .group_to_apply_to_selected_step_base import GroupToApplyToSelectedStepBase
from .pipeline_step_base import PipelineStepBase


class DataGroupInPathAppliedStep(GroupToApplyToSelectedStepBase):
    '''Apply a contained processing step to the sub-tree rooted at a given path.'''

    def __init__(
        self,
        processing_step_to_apply: PipelineStepBase,
        path_to_apply_to: Union[str, int, Sequence[str, int]],
    ):
        '''

        Args:
            processing_step_to_apply: The contained processing step
            path_to_apply_to: Path to the root of the sub-tree to apply `processing_step_to_apply` to
        '''

        GroupToApplyToSelectedStepBase.__init__(self, processing_step_to_apply)
        self._path_to_apply_to = path_to_apply_to

    @override
    def _check_and_get_paths_to_apply_to(self, data: SampleDataGroup) -> Sequence[Sequence[str, int]]:

        if not data.path_exists_and_is_data_group_field(self._path_to_apply_to):
            raise ValueError(
                f"DataGroupInPathAppliedStep: Path `{self._path_to_apply_to}` does not exist or is not a data group field."
            )
        if data.path_is_single_name(self._path_to_apply_to):
            path = (self._path_to_apply_to,)
        else:
            path = self._path_to_apply_to
        paths = (path,)
        return paths
