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

import torch

# Import the batching-helpers package
import accvlab.batching_helpers as batching_helpers

# Import the matcher and loss computation modules (parts of the example implementation)
from matcher import Matcher
from loss_computation import LossComputation

# Import the example input data (helper for running the example)
import input_data


def loss_computation_main(rects_gt, classes_gt, rects_pred, classes_pred, pred_existence, weights_gt):

    # ===== Step 1: Conversion of the GT per-sample data to RaggedBatch instances =====

    # @NOTE
    # Typically, the ground truth (GT) is provided as a list containing per-sample GT data as individual
    # tensors. Here, this format is converted into RaggedBatch objects containing the whole batch.
    # Note that except for the first call, a `other_with_same_sample_sizes` parameter is present. This
    # is optional, but saves memory by re-using the `mask` and `sample_sizes` (see `RaggedBatch`
    # documentation) of the first created instance. This is possible as all the GT data refers to the same
    # objects, so that for a given sample, the number of objects is the same for the different types of GT
    # data.
    rects_gt_compact = batching_helpers.combine_data(rects_gt)
    classes_gt_compact = batching_helpers.combine_data(
        classes_gt, other_with_same_sample_sizes=rects_gt_compact
    )
    weights_gt_compact = batching_helpers.combine_data(
        weights_gt, other_with_same_sample_sizes=rects_gt_compact
    )

    # ===== Step 2: Matching of the predictions to the GT objects =====

    # @NOTE
    # Get the matches for the individual samples. `matched_gt_indices` and `matched_pred_indices` contain
    # indices for matches for the GT and predictions, respectively. As each sample contains a different number
    # of matches, `RaggedBatch` instances are used to store the indices for both the GT and the predictions.
    matcher = Matcher()
    matched_gt_indices, matched_pred_indices = matcher(
        rects_gt_compact, classes_gt_compact, rects_pred, classes_pred
    )

    # ===== Step 3: The actual loss computation =====

    # @NOTE
    # Compute the actual loss given GT and prediction data, as well as the matches established by the matcher.
    loss_comp = LossComputation()
    per_sample_loss = loss_comp(
        rects_gt_compact,
        classes_gt_compact,
        rects_pred,
        classes_pred,
        pred_existence,
        weights_gt_compact,
        matched_gt_indices,
        matched_pred_indices,
    )

    # @NOTE
    # The loss computation returns per-sample losses, and they can be used as such after the computation
    # (e.g. logged, weighted, etc.). Here, we just sum the per-sample losses to obtain the final loss.
    final_loss = torch.sum(per_sample_loss)
    return final_loss


if __name__ == "__main__":
    loss = loss_computation_main(
        input_data.rects_gt,
        input_data.classes_gt,
        input_data.rects_pred,
        input_data.classes_pred_onehot,
        input_data.pred_existence,
        input_data.weights_gt,
    )
    print(f"Loss: {loss}")
