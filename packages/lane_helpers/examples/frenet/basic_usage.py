# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch

from accvlab.lane_helpers import frenet


def main() -> None:
    # @NOTE Define an ordered 2D reference lane on CUDA.
    reference_lane_points = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
        device="cuda",
        dtype=torch.float32,
    )

    # @NOTE Define Cartesian points on both sides of the reference lane.
    points_to_transform = torch.tensor(
        [[0.5, 1.0], [1.5, -0.5], [2.5, 2.0], [3.5, -1.0]],
        device="cuda",
        dtype=torch.float32,
    )

    # @NOTE Transform the points using normals derived from the lane geometry.
    frenet_points = frenet.transform(reference_lane_points, points_to_transform)

    # @NOTE The output columns contain longitudinal s and signed lateral d.
    print(f"Frenet points (s, d):\n{frenet_points}")


if __name__ == "__main__":
    main()
