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


class LossComputation:

    def __call__(
        self,
        bboxes_gt,
        classes_gt,
        bboxes_pred,
        classes_pred,
        existence_pred,
        weights_gt,
        matches_gt,
        matches_pred,
    ):
        # @NOTE
        # Extract matched ground truth and prediction data using the indices from matching.
        # This creates element-wise correspondences between GT and prediction objects,
        # enabling direct comparison in subsequent loss computations.
        # See `batching_helpers.batched_indexing_access()` documentation for details.
        cls_gt_matched = batching_helpers.batched_indexing_access(classes_gt, matches_gt).to_dtype(
            torch.int64
        )
        cls_pred_matched = batching_helpers.batched_indexing_access(classes_pred, matches_pred)
        bbxs_gt_matched = batching_helpers.batched_indexing_access(bboxes_gt, matches_gt)
        bbxs_pred_matched = batching_helpers.batched_indexing_access(bboxes_pred, matches_pred)
        weights_matched = batching_helpers.batched_indexing_access(weights_gt, matches_gt)

        # @NOTE
        # Compute (per-object) losses. Note that this is a batched operation and furthermore, that the
        # loss functions themselves are not specifically implemented for non-uniform batches and do not
        # distinguish between actual objects and filler entries in the data. This means that
        # in a real use-case, already available batched loss functions can be readily re-used.
        #
        # Also, please note that the loss functions do not reduce over the individual objects/targets.
        # This enables us to wrap the per-object losses as `RaggedBatch` instances and use the
        # `RaggedBatch` and `batching-helpers` functionality to handle the non-uniform sample sizes (e.g.
        # when summing/averaging over the valid entries only).
        #
        # Note that other ways of handling the padded entries are also possible if the loss functions do
        # reduce over the objects. One possible way is to provide appropriate (0.0) weights for the padded
        # entries (however, be cautious of potential NaN values when using this approach).
        class_per_object_loss_data = self._per_object_class_l1_loss_labels_gt(
            cls_gt_matched.tensor, cls_pred_matched.tensor, weights_matched.tensor
        )
        bbox_per_object_loss_data = self._per_object_bbox_overlap_loss(
            bbxs_gt_matched.tensor, bbxs_pred_matched.tensor, weights_matched.tensor
        )

        # @NOTE
        # Wrap the per-object losses as `RaggedBatch` instances. Similarly to the cost matrices in the
        # matcher, this can be done as the filler elements in the loss tensors are located where the
        # `RaggedBatch` implementation expects them (as the filler locations in the loss computation inputs
        # were defined by the `RaggedBatch` instances containing the input data, and no permutations of
        # objects are performed in the loss computation).
        class_per_object_loss = cls_gt_matched.create_with_sample_sizes_like_self(
            class_per_object_loss_data, non_uniform_dim=1
        )
        bbox_per_object_loss = bbxs_gt_matched.create_with_sample_sizes_like_self(
            bbox_per_object_loss_data, non_uniform_dim=1
        )

        # @NOTE
        # Sum up loss for the individual objects. As the loss functions do not explicitly handle the padded
        # entries, the loss computation is also performed for those. This means that the filler entries may
        # contain non-zero values (including `NaN`). Therefore, the filler values would potentially influence
        # the sum if taken into consideration. This means we cannot use `torch.sum()` directly. Instead, we
        # use the `sum_over_targets()` function provided by the `batching-helpers` package.
        class_loss = batching_helpers.sum_over_targets(class_per_object_loss)
        bbox_loss = batching_helpers.sum_over_targets(bbox_per_object_loss)

        # @NOTE
        # Compute existence loss next. This loss is different from the other losses in that the computation is
        # done for all predictions, not only the matched ones.

        # @NOTE
        # First, create a mask which is `True` for existing (matched) targets and `False` for non-existent
        # ones. The mask is created from the indices of the matched predictions (also see the
        # `batching_helpers.get_mask_from_indices()` documentation).
        existence_mask = batching_helpers.get_mask_from_indices(existence_pred.shape[1], matches_pred)

        # @NOTE
        # Additionally, compute the overlap (`1.0 - bbox_per_target_loss`) and use it as a weight in the
        # existence loss (in combination with the weights from `weights_gt`) as follows:
        #   - Use the so computed weights directly for the matched objects
        #   - Compute average value and use it for the non-matched objects
        # In addition, apply a compensation factor between existing and non-existing objects to the
        # non-matched objects in order to account for the imbalance.
        #
        # To obtain the overall weights used for all predictions, the following steps are performed:
        # 1. Compute the overlap weights for the matched objects (from `bbox_per_object_loss`)
        # 2. Combine the overlap weights with `weights_matched` (which contains the values from `weights_gt`
        #    for the matched objects) to obtain `existence_weights_matched`.
        # 3. Map the resulting `existence_weights_matched` back to all predictions & also set the weights for
        #    non-existent (i.e. non-matched) predictions in the process. This is done as follows:
        #   a) First compute the per-sample mean values of `existence_weights_matched` (averaging over the
        #      existing objects) to obtain `weights_means`.
        #   b) Then, compute per-sample `imbalance_factors` compensating for the imbalance between existing
        #      and non-existing objects.
        #   c) Multiply the `weights_means` with `imbalance_factors` to obtain `weights_mean_adjusted`.
        #   d) Initialize `existence_weights_preds` (which contains the weights for all predictions and is of
        #      corresponding shape) with the values from `weights_mean_adjusted`. These initial values are the
        #      weights for the non-matched predictions.
        #   e) Write the values from `existence_weights_matched` into
        #      `batching_helpers.batched_indexing_write()` for the matched predictions (i.e. use the weights
        #      in `existence_weights_matched` for those), while leaving the other values (i.e. non-matched)
        #      unchanged.
        #
        # The points above are implemented as follows:

        # @NOTE
        # 1. Compute the overlap weights for the matched bboxes (from `bbox_per_object_loss`).
        #
        # Note the use of the `apply()` convenience method to apply a function to the data tensor (i.e.
        # `tensor`) of the `RaggedBatch` instance. The line:
        #   >>> overlap_weights_matched = bbox_per_object_loss.apply(lambda tensor: 1.0 - tensor)
        # is equivalent to:
        #   >>> tensor = bbox_per_object_loss.tensor
        #   >>> tensor = 1.0 - tensor
        #   >>> overlap_weights_matched = bbox_per_object_loss.create_with_sample_sizes_like_self(tensor)
        # Note that the `apply()` method returns a new `RaggedBatch` instance. Also, the passed function
        # may accept more than one argument, in which case `sample_sizes` and `mask` are also passed to
        # the function (but should not be modified). Please refer to the documentation of
        # `RaggedBatch.apply()` for more details.
        overlap_weights_matched = bbox_per_object_loss.apply(lambda tensor: 1.0 - tensor)

        # @NOTE
        # 2. Combine the overlap weights with `weights_matched` (which contains the values from `weights_gt`
        #    for the matched objects).
        #
        # Note that here, data tensors of two `RaggedBatch` instances are processed in the lambda function.
        # As both `RaggedBatch` instances represent the same sample sizes and non-uniform dimension, it does
        # not matter which one calls the `apply()` method and for which one the data tensor is accessed as
        # `.tensor`.
        existence_weights_matched = weights_matched.apply(
            lambda tensor: tensor * overlap_weights_matched.tensor
        )

        # @NOTE
        # 3a). First compute the per-sample mean values of `existence_weights_matched` (averaging over the
        #      existing objects) to obtain `weights_means`.
        #
        # As the target dimension is padded, `torch.mean()` cannot be used both
        #   - for the reasons discussed above for summation over objects (i.e. the number of actual objects
        #     does not necessarily correspond to the tensor size)
        #   - because `torch.mean()` would divide the sum by a wrong number of elements for samples containing
        #     filler elements
        # Instead, we use the method provided by the `batching-helpers` package:
        weights_means = batching_helpers.average_over_targets(existence_weights_matched)

        # @NOTE
        # 3b). Then, compute per-sample `imbalance_factors` compensating for the imbalance between existing
        #      and non-existing objects.
        #
        # First, obtain the number of predictions.
        num_preds = bboxes_pred.shape[1]
        # @NOTE
        # Then, compute the imbalance correction factor as follows:
        #   - Divide by the number of non-existent targets
        #     (i.e. `num_preds - overlap_weights_matched.sample_sizes`)
        #   - Multiply by the number of existing targets (i.e. `overlap_weights_matched.sample_sizes`)
        # Note that the `nan_to_num()` function is used to handle the case where the number of non-existent
        # targets is zero.
        imbalance_factors = torch.nan_to_num(
            overlap_weights_matched.sample_sizes / (num_preds - overlap_weights_matched.sample_sizes), 0.0
        )

        # @NOTE
        # 3c). Multiply the `weights_means` with `imbalance_factors` to obtain `weights_mean_adjusted`.
        weights_mean_adjusted = weights_means * imbalance_factors

        # @NOTE
        # 3d). Initialize `existence_weights_preds` (which contains the weights for all predictions and is of
        #      corresponding shape) with the values from `weights_mean_adjusted`. These initial values are the
        #      weights for the non-matched predictions.
        existence_weights_preds = weights_mean_adjusted.unsqueeze(-1).repeat(1, classes_pred.shape[1])

        # @NOTE
        # 3e). Write the values from `existence_weights_matched` into `existence_weights_preds` for the
        #      matched predictions (i.e. use the weights in `existence_weights_matched` for those), while
        #      leaving the other values unchanged.
        #
        # Note that the `batched_indexing_write()` function is equivalent to `__setitem__()` for the unbatched
        # (single-sample) case using the build-in tensor indexing operator.
        existence_weights_preds = batching_helpers.batched_indexing_write(
            existence_weights_matched, matches_pred, existence_weights_preds
        )

        # @NOTE
        # Compute existence loss (considering all predictions, not only the matched ones).
        # Note that the loss has uniform size, and therefore we can directly use `torch.sum()`
        # to sum over the objects.
        existence_per_object_loss = self._per_object_existence_loss(
            existence_pred, existence_mask, existence_weights_preds
        )
        existence_loss = torch.sum(existence_per_object_loss, 1)

        # @NOTE
        # Sum up all losses & return result.
        loss = class_loss + bbox_loss + existence_loss
        return loss

    # Example loss function for the loss computation. This is not the focus of the example.
    @staticmethod
    def _per_object_class_l1_loss_labels_gt(classes_gt, classes_pred, weights):

        def per_object_class_l1_loss_one_hot_gt(classes_gt, classes_pred, weights):

            diff = torch.abs(classes_gt - classes_pred)
            weighted_diff = weights.unsqueeze(-1) * diff

            # Compute the sum over the classes
            res = torch.sum(weighted_diff, dim=2)

            return res

        # Note: This part of the loss computation is not batched. However, we do not focus on loss
        # function implementation here and in a practical application, the loss can be implemented
        # differently (e.g. custom PyTorch extension).
        num_classes = classes_pred.shape[-1]
        batch_size = classes_gt.shape[0]
        res = [None] * batch_size
        for s, gt in enumerate(classes_gt):
            res_s = torch.zeros((gt.shape[0], num_classes), dtype=torch.float32, device=gt.device)
            for i, label in enumerate(gt):
                res_s[i, label] = 1.0
            res[s] = res_s
        classes_gt_one_hot = torch.stack(res, dim=0)
        # end of non_batched part
        res = per_object_class_l1_loss_one_hot_gt(classes_gt_one_hot, classes_pred, weights)
        return res

    # Example batched loss function. It is used in the example, but the implementation
    # of this function is not the focus of the example.
    @staticmethod
    def _per_object_bbox_overlap_loss(bboxes_gt, bboxes_pred, weights, eps=1e-6):
        areas_gt = torch.prod(bboxes_gt[..., 2:4] - bboxes_gt[..., 0:2], axis=-1)
        areas_pred = torch.prod(bboxes_pred[..., 2:4] - bboxes_pred[..., 0:2], axis=-1)

        rects_gt_ul = bboxes_gt[..., 0:2]
        rects_gt_lr = bboxes_gt[..., 2:4]
        rects_pred_ul = bboxes_pred[..., 0:2]
        rects_pred_lr = bboxes_pred[..., 2:4]

        intersections_ul = torch.max(rects_gt_ul, rects_pred_ul)
        intersections_lr = torch.min(rects_gt_lr, rects_pred_lr)
        sizes_intersections = intersections_lr - intersections_ul
        sizes_intersections[sizes_intersections < 0.0] = 0.0
        areas_intersections = torch.prod(sizes_intersections, axis=-1)

        areas_union = areas_gt + areas_pred - areas_intersections
        areas_union[areas_union < eps] = eps

        target_loss = 1.0 - areas_intersections / areas_union

        target_loss = target_loss * weights

        return target_loss

    # Example batched loss function. It is used in the example, but the implementation
    # of this function is not the focus of the example.
    @staticmethod
    def _per_object_existence_loss(existence_pred, existence_mask, weights):
        existence_gt = existence_mask.to(dtype=torch.float32)
        diff = torch.abs(existence_pred - existence_gt)
        loss = weights * diff
        return loss
