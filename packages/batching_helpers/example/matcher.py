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
import accvlab.batching_helpers as batching_helpers
from scipy.optimize import linear_sum_assignment


class Matcher:

    def __call__(self, rects_gt, classes_gt, rects_pred, classes_pred):
        # @NOTE
        # Get the cost matrices denoting the cost for each GT to prediction combination. Note that as the
        # samples in the GT data are padded to uniform size (see documentation of `RaggedBatch.tensor`), the
        # same will be true for the matrices.
        batch_size = rects_gt.shape[0]
        iou_cost_matrices = self._iou_cost_func(rects_gt.tensor, rects_pred)
        class_cost_matrices = self._class_l1_cost_func_gt_labels(classes_gt.tensor, classes_pred)
        total_cost_matrices = iou_cost_matrices + class_cost_matrices

        # @NOTE
        # During cost matrix computation, we assume uniform batch size (and use filler values). However, the
        # valid cost matrices are non-uniform in size. Along `dim==2` (iterating over the GT objects), the
        # sample sizes correspond to the sample sizes of the GT inputs (there, along `dim==1`). Create a
        # RaggedBatch containing the matrices. Note that this will correctly handle the filler regions in the
        # matrices, as they exactly correspond to the format used in `RaggedBatch.tensor`. This is as follows:
        #   - In the input data to the matrix computations originally from `RaggedBatch` instances, the filler
        #     values are in the correct format (i.e. always after the valid data)
        #   - The matrix computations do not perform any permutations of the data, so that the filler values
        #     remain in the same locations (but along a different dimension)
        total_cost_matrices = classes_gt.create_with_sample_sizes_like_self(
            total_cost_matrices, non_uniform_dim=2
        )

        # @NOTE
        # The Hungarian matching is done on the CPU one sample at a time. Therefore, move the data to the CPU
        # and split RaggedBatch instances containing the cost matrices into individual samples. Note that
        # `split()` removes the filler value padding, so that the valid matrices with correct sample sizes are
        # obtained.
        device_cpu = torch.device("cpu")
        total_cost_matrices_cpu = total_cost_matrices.to_device(device_cpu)
        total_cost_matrices_list = total_cost_matrices_cpu.split()

        # @NOTE: Perform matching for each sample
        matched_gt_index_list = [None] * batch_size
        matched_pred_index_list = [None] * batch_size
        for i, cost_mat in enumerate(total_cost_matrices_list):
            m_pred, m_gt = linear_sum_assignment(cost_mat)
            matched_gt_index_list[i] = torch.tensor(m_gt, dtype=torch.int64, device=device_cpu)
            matched_pred_index_list[i] = torch.tensor(m_pred, dtype=torch.int64, device=device_cpu)

        # @NOTE
        # Combine resulting indices for the individual samples into RaggedBatch instances representing the
        # whole batch.
        matched_gt_indices = batching_helpers.combine_data(matched_gt_index_list)
        matched_pred_indices = batching_helpers.combine_data(
            matched_pred_index_list, other_with_same_sample_sizes=matched_gt_indices
        )
        # @NOTE: Move results to the GPU
        matched_gt_indices = matched_gt_indices.to_device(device=rects_gt.device)
        matched_pred_indices = matched_pred_indices.to_device(device=rects_gt.device)

        return matched_gt_indices, matched_pred_indices

    # Example batched cost function for the matcher. It is used in the example, but the implementation
    # of this function is not the focus of the example.
    @staticmethod
    def _iou_cost_func(rects_gt, rects_pred, eps=1e-6):

        # With broadcasting, using the `_ext` variants will lead to pair-wise results for all possible
        # combinations
        rects_gt_ext = rects_gt.unsqueeze(1)
        rects_pred_ext = rects_pred.unsqueeze(2)

        areas_gt = torch.prod(rects_gt_ext[..., 2:4] - rects_gt_ext[..., 0:2], axis=-1)
        areas_pred = torch.prod(rects_pred_ext[..., 2:4] - rects_pred_ext[..., 0:2], axis=-1)

        rects_gt_ul = rects_gt_ext[..., 0:2]
        rects_gt_lr = rects_gt_ext[..., 2:4]
        rects_pred_ul = rects_pred_ext[..., 0:2]
        rects_pred_lr = rects_pred_ext[..., 2:4]

        intersections_ul = torch.max(rects_gt_ul, rects_pred_ul)
        intersections_lr = torch.min(rects_gt_lr, rects_pred_lr)
        sizes_intersections = intersections_lr - intersections_ul
        sizes_intersections[sizes_intersections < 0.0] = 0.0
        areas_intersections = torch.prod(sizes_intersections, axis=-1)

        areas_union = areas_gt + areas_pred - areas_intersections
        areas_union[areas_union < eps] = eps

        res = 1.0 - areas_intersections / areas_union

        return res

    # Example batched cost function for the matcher. It is used in the example, but the implementation
    # of this function is not the focus of the example.
    @staticmethod
    def _class_l1_cost_func_gt_labels(classes_gt, classes_pred_one_hot):

        # Internal helper function
        def class_l1_cost_func_gt_one_hot(classes_gt_one_hot, classes_pred_one_hot):
            prod = torch.einsum('bik,bjk->bij', classes_pred_one_hot, classes_gt_one_hot)
            cost = 1.0 - prod
            return cost

        # Note: This part of the loss computation is not computed in a batched manner. However, this
        # is not the focus of the example and in an actual application, the loss can be implemented
        # differently (e.g. custom extension).
        num_classes = classes_pred_one_hot.shape[-1]
        batch_size = classes_gt.shape[0]
        res = [None] * batch_size
        for s, gt in enumerate(classes_gt):
            res_s = torch.zeros((gt.shape[0], num_classes), dtype=torch.float32, device=gt.device)
            for i, cls in enumerate(gt):
                res_s[i, cls] = 1.0
            res[s] = res_s
        classes_gt_one_hot = torch.stack(res, dim=0)
        # end of non_batched part
        cost = class_l1_cost_func_gt_one_hot(classes_gt_one_hot, classes_pred_one_hot)
        return cost
