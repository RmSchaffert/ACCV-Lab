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

from typing import Sequence, Union

import numpy as np

from nvidia.dali import fn


class DaliFakeRandomGenerator:

    class RangeReplacement:
        def __init__(self, val_range: Union[Sequence[float], None], val_sequence: Sequence[float]):
            self.val_sequence = val_sequence
            self.curr_idx = 0
            self.val_range = val_range

    def __init__(self, sequences: Sequence[RangeReplacement]):
        self._sequences = sequences
        self._ranges = []

    def _get_range_idx(self, range_in):
        def range_close(range1, range2, epsilon=1e-6):
            return np.allclose(np.array(range1), np.array(range2), atol=epsilon)

        for i, seq in enumerate(self._sequences):
            if seq.val_range is None:
                continue
            if range_close(seq.val_range, range_in):
                return i

        for i, seq in enumerate(self._sequences):
            if seq.val_range is None:
                return i

        raise ValueError(
            f"No range replacement found for range {range_in}. Add a range replacement for "
            f"this range or a default (None) range replacement."
        )

    def _get_next(self, range):
        index = self._get_range_idx(range)
        val_to_ret = self._sequences[index].val_sequence[
            self._sequences[index].curr_idx % len(self._sequences[index].val_sequence)
        ]
        self._sequences[index].curr_idx += 1
        res = np.array(val_to_ret, dtype=np.float32)
        return res

    def _get_process_func(self):
        # Inner function used to enclose `self`.
        def process_func(range, **kwargs):
            self._ranges.append(range)
            return self._get_next(range)

        return process_func

    def _to_call(self, range, **kwargs):
        res = fn.python_function(range, function=self._get_process_func(), num_outputs=1)
        return res

    def get_generator(self):
        # Inner function used to enclose `self`.
        def generator_func(range, **kwargs):
            return self._to_call(range, **kwargs)

        return generator_func

    def get_used_ranges(self):
        return self._ranges


if __name__ == "__main__":

    from nvidia.dali import fn
    from nvidia.dali import pipeline_def

    sequences = [
        DaliFakeRandomGenerator.RangeReplacement(None, [1.0, 2.0, 3.0]),
        DaliFakeRandomGenerator.RangeReplacement([0.0, 1.0], [4.0, 5.0, 6.0]),
        DaliFakeRandomGenerator.RangeReplacement([0.3, 5.0], [7.0, 8.0, 9.0]),
    ]
    generator = DaliFakeRandomGenerator(sequences)

    fn.random.uniform = generator.get_generator()

    @pipeline_def
    def test_pipeline():
        data = fn.random.uniform(range=[0.0, 1.0])
        data2 = fn.random.uniform(range=[1.0, 2.0])
        data3 = fn.random.uniform(range=[0.3, 5.0])
        data4 = fn.random.uniform(range=[0.3, 5.0])
        return data, data2, data3, data4

    pipe = test_pipeline(batch_size=1, num_threads=1, device_id=0)
    pipe.build()
    res = pipe.run()
    print(res[0].as_cpu().as_array())
    print(res[1].as_cpu().as_array())
    print(res[2].as_cpu().as_array())
    print(res[3].as_cpu().as_array())
    print(generator.get_used_ranges())
