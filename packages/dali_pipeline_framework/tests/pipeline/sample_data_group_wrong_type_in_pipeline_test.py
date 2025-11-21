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

import pytest

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np

import nvidia.dali.fn as fn
from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import PipelineStepBase


class _WrongTypeSetter(PipelineStepBase):

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # Field 'value' is declared FLOAT; set an INT32 DataNode to trigger type check failure
        # Provide an INT32 DataNode (list of ints is required for idata)
        wrong = fn.constant(idata=[1, 2, 3], shape=(3,), dtype=DALIDataType.INT32)
        data["value"] = wrong
        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        # Schema does not change; keep declared type
        return data_empty


class _Provider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        # Provide FLOAT data (vector to avoid scalars)
        res["value"] = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        return res

    @override
    def get_number_of_samples(self) -> int:
        return 4

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        s = SampleDataGroup()
        s.add_data_field("value", DALIDataType.FLOAT)
        return s


@pytest.mark.parametrize("check_data_format", [True, False])
def test_wrong_type_assignment_check_in_pipeline(check_data_format: bool):
    provider = _Provider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = _WrongTypeSetter()

    # Error is raised. Note that the error is allowed to be raised anywhere between the pipeline definition
    # and getting the first element from the iterator. In practice, the error is raised when the iterator is
    # created.
    if check_data_format:
        with pytest.raises(RuntimeError, match="Expected type .*float32.* actual data type .*int32"):
            pipeline_def = PipelineDefinition(
                data_loading_callable_iterable=input_callable,
                preprocess_functors=[step],
                check_data_format=True,
            )

            pipeline = pipeline_def.get_dali_pipeline(
                enable_conditionals=True,
                batch_size=1,
                prefetch_queue_depth=1,
                num_threads=1,
            )

            iterator = DALIStructuredOutputIterator(
                2, pipeline, pipeline_def.check_and_get_output_data_structure()
            )

            it = iter(iterator)
            _ = next(it)
    else:
        pipeline_def = PipelineDefinition(
            data_loading_callable_iterable=input_callable,
            preprocess_functors=[step],
            check_data_format=False,
        )
        pipeline = pipeline_def.get_dali_pipeline(
            enable_conditionals=True,
            batch_size=1,
            prefetch_queue_depth=1,
            num_threads=1,
        )
        iterator = DALIStructuredOutputIterator(
            2, pipeline, pipeline_def.check_and_get_output_data_structure()
        )

        it = iter(iterator)
        _ = next(it)


if __name__ == "__main__":
    pytest.main([__file__])
