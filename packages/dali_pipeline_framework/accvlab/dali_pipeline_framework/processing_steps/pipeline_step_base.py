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

from abc import ABC, abstractmethod

from ..pipeline.sample_data_group import SampleDataGroup


class PipelineStepBase(ABC):
    '''Base class for pipeline processing steps.

    Pipeline processing steps are the building blocks of the pipeline and represent individual
    operations applied to input data in sequence to produce outputs.

    Provides the common interface and common functionality shared by all processing steps:

      - Checking the input data format for compatibility and setting the output data format (blueprint)
        (see :meth:`check_input_data_format_and_set_output_data_format`).
      - Applying the step via :meth:`__call__`. This

        - Invokes :meth:`_process` to perform the actual processing.
        - Validates the resulting data format against a reference blueprint from
          :meth:`check_input_data_format_and_set_output_data_format` to ensure that the resulting format is
          "as advertised", i.e. as obtained by independent calls to
          :meth:`check_input_data_format_and_set_output_data_format`. Note that this check is performed at
          DALI graph construction time and therefore does not affect runtime during training.

      - Support for operating on sub-trees of input data (through specialized wrapper steps, see
        :class:`GroupToApplyToSelectedStepBase`).

    .. _pipeline_step_base_access_modifier_wrapper_discussion:

    .. admonition:: Consistent & Independent Data Processing
        :class: important

        Many of the included processing steps can be configured to operate on more than one field in the input
        :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup` object. For some
        steps (e.g. those which apply random transformations), the question arises whether these steps should
        apply consistent processing across all fields they process (e.g. same augmentation transformation for
        all images), or if the processing should happen independently for different fields (e.g. different
        transformations for different images). The answer to this question depends on the use-case.

        By default, the processing steps are designed to apply consistent processing. For example,
        :class:`AffineTransformer` applies the same spatial transform to all processed images, as well as
        corresponding fields such as point sets defined on the image or projection matrices. This ensures
        that:

          - Consistent randomization is possible if needed (e.g., between an image, a corresponding
            segmentation mask, projection matrix, and points defined on the image).
          - No correspondences between multiple fields need to be explicitly maintained. For example, if
            multiple images and projection matrices are present, there is no need to know which projection
            matrix corresponds to which image, as the same transformation is applied to all of them. This is
            useful when processing multiple fields which are related to one another.

        To ensure that independent processing (e.g. different randomizations) can be applied to different
        parts of the data (e.g., different randomizations for data from different cameras), sub-classes of
        :class:`GroupToApplyToSelectedStepBase` can be used to select one or more parts (sub-trees) of the
        input data to process independently of each other. The selection of the sub-trees also allows to
        establish field correspondences (e.g., process the image and projection matrix from one camera
        consistently) in a natural way, i.e. by grouping all related fields in one sub-tree (e.g. one sub-tree
        per camera).

        The available wrappers include :class:`DataGroupInPathAppliedStep`,
        :class:`DataGroupsWithNameAppliedStep`, :class:`DataGroupArrayInPathElementsAppliedStep`, and
        :class:`DataGroupArrayWithNameElementsAppliedStep`. Please see the documentation of these classes for
        more details. If necessary, new wrappers can be added by subclassing
        :class:`GroupToApplyToSelectedStepBase`.

        Having both options (e.g. consistent or different randomizations for different parts of the data)
        available, as well as the ability to group related data (e.g. all images and projection matrices for
        one camera) allows for flexible pipeline design which can be tailored to the specific use-case by
        configuration.

    .. automethod:: PipelineStepBase._check_and_adjust_data_format_input_to_output


    .. automethod:: PipelineStepBase._process

    '''

    def __call__(self, data: SampleDataGroup) -> SampleDataGroup:
        '''Apply the processing step and validate its output format.

        Important:
            To define the actual functionality of a processing step, override :meth:`_process`, not
            this method.

        Args:
            data: Input data to process.

        Returns:
            Processed output data.
        '''

        # Note that this needs to be done before calling `_process()`, as `_process()` may change `data`.
        data_plueprint_in = data.get_empty_like_self()

        processed = self._process(data)

        # The check is performed at DALI graph construction time and therefore does not affect runtime during training.
        reference_blueprint = self.check_input_data_format_and_set_output_data_format(data_plueprint_in)
        if not processed.type_matches(reference_blueprint):
            raise AssertionError(
                f"SampleDataGroup data format returned by the _process function does not match the reference (returned by `check_input_data_format_and_set_output_data_format()`).\n ##### From `_process()`:\n {processed}\n ##### Reference:\n {reference_blueprint}\n ##########"
            )

        return processed

    @abstractmethod
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        '''Apply the processing step to the input, or to a selected sub-tree when wrapped accordingly.

        Individual processing steps need to override this method and implement the actual functionality.

        The method may mutate the input data; callers must not rely on the input remaining unchanged or
        corresponding to the output after the call.

        Note:

            - Override this method in each (non-abstract) derived class to define the actual functionality.
            - This method is called by :meth:`__call__` and should not be called directly.

        Args:
            data: Data to be processed by the step.

        Returns:
            Resulting processed data.

        '''
        pass

    def check_input_data_format_and_set_output_data_format(
        self, data_empty: SampleDataGroup
    ) -> SampleDataGroup:
        '''Check the input data format for compatibility and return the output data format (blueprint).

        Compatibility typically means that expected data fields are present and types are compatible, and
        that the output data fields can be added (are not already present). Typical changes to the data
        format include additions/removals of fields or changes to data types (e.g., an image may change
        from ``types.DALIDataType.UINT8`` to ``types.DALIDataType.FLOAT`` in a normalization step).

        This method does not modify ``data_empty`` in place; it returns a new :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup`
        describing the modified format.

        If the input data format is incompatible, an exception is raised.

        Important:
            To define the actual functionality of the check, override
            :meth:`_check_and_adjust_data_format_input_to_output`, not this method.

        Args:
            data_empty: Input data format (blueprint),

        Returns:
            Resulting data format (blueprint).

        '''

        # Do not modify the original SampleDataGroup object. `get_empty_like_self()` returns a copy without
        # data for use as a 'blueprint' describing the data format. Using this method instead of `get_copy()`
        # makes this intent explicit.
        data_empty = data_empty.get_empty_like_self()

        data_empty_res = self._check_and_adjust_data_format_input_to_output(data_empty)

        return data_empty_res

    @abstractmethod
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:
        '''Check the input data format for compatibility and return the output data format (blueprint).

        If the input data format is incompatible, raise an exception describing the problem.

        Please see :meth:`check_input_data_format_and_set_output_data_format` for a description of typical
        checks and format changes that need to be performed here.

        This method may or may not modify ``data_empty`` directly, but in any case has to return an object
        representing the modified format (i.e., either the modified ``data_empty`` or a new object).

        Note:

            - Override this method in each (non-abstract) derived class to define the actual functionality.
            - This method is called by :meth:`check_input_data_format_and_set_output_data_format` and should
              not be called directly.


        Args:
            data_empty: Input data format (blueprint)

        Returns:
            Resulting data format (blueprint)

        '''
        pass
