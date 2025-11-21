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

from typing import Union

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import nvidia.dali.fn as fn
import nvidia.dali.types as types

from ..pipeline.sample_data_group import SampleDataGroup

from ..internal_helpers.mini_parser import (
    Parser,
    AST,
    Literal,
    Variable,
    Comparison,
    Or,
    And,
    Not,
    UnaryMinus,
)

from .pipeline_step_base import PipelineStepBase


class AnnotationElementConditionEval(PipelineStepBase):
    '''Evaluate a declarative condition per annotation element and store the boolean result.

    This step looks for data group fields (see documentation of :class:`SampleDataGroup`) corresponding to
    annotations, and applies the the defined conditions to data fields inside the annotation. The results are
    stored as a new data field inside the annotation. Both the data fields used in the condition and the
    resulting data fields are referenced by name in the condition string.

    The used fields are expected to be 1D sequences (one value per object). The condition is evaluated
    per element, producing a boolean sequence with one result per object.

    Note:
        While 1D sequences are expected, the data may be formatted as 2D tensors. In this case, one
        dimension needs to have a size of 1.

    The condition must start with a variable name, followed by an assignment operator, followed by an
    expression.

    The expression can contain variables (will be mapped to the data fields of the annotation), literals, and
    operators.

    The supported operators are:
      - Logical operators: ``or``, ``and``, ``not``
      - Comparison operators: ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
      - Parentheses: ``(`` and ``)``
      - Unary minus: ``-``; e.g. ``-_b1 < -10.5`` is valid.
      - Assignment operator: ``=``

    The syntax is similar to Python. However, note that
      - Only the operators defined above are supported.
      - Direct comparisons of more than two values are not supported (e.g. ``a < b < c`` is not supported).
      - Only numeric literals are supported. ``True`` and ``False`` are not supported (not needed; use
        negation instead of comparison to ``False``).

    The result of the condition is stored in a new data field, which is added to the annotation. The name of
    the result data field is also defined inside the condition string.

    Example:

        The condition can be described in a syntax similar to Python, e.g.:
          ``is_valid = (num_lidar_points >= 1 or num_radar_points >= 1) and visibility_levels > 0 and category > 0``

        In this case:

          - The data fields ``num_lidar_points`` and ``num_radar_points``, ``visibility_levels``, and ``category``
            are expected to be children of the annotation data group field.
          - The result of the condition is stored in a new data field inside the annotation data group field,
            named ``is_valid``.

    Important:
        In order to use data fields inside the condition, their names must follow the rules of Python
        variable names (e.g. no spaces, no special characters, do not start with a digit).

    See also:
        - Specific complex conditions can be checked with :class:`VisibleBboxSelector`,
          :class:`PointsInRangeCheck`, and the results of these checks can be combined with this step.
        - :class:`ConditionalElementRemoval` can be used to remove elements from the data based on this
          condition.
        - :class:`BoundingBoxToHeatmapConverter` has both input and output fields containing boolean
          masks denoting the active objects.

    Args:
        annotation_field_name:
            Name of annotation data group field. Note that there can be more than one annotation field (e.g.
            one for objects visible in each camera). In this case, these annotations are all processed
            (independently of each other).
        condition:
            Condition to be applied. Please see the description above for more details.
        remove_data_fields_used_in_condition:
            Whether to remove the data fields used in the condition after evaluating the condition.
            This is a convenience feature and can be set to `True` if the data fields are not used after
            evaluating the condition. However, note that if some of the data fields are used, it has to be
            set to `False`, as the fields are not available after this step otherwise.
    '''

    def __init__(
        self,
        annotation_field_name: Union[str, int],
        condition: str,
        remove_data_fields_used_in_condition: bool,
    ):
        self._annotation_field_name = annotation_field_name
        self._condition_statement = Parser(condition).parse()
        self._condition = self._condition_statement.expression
        self._result_field_name = self._condition_statement.variable.name
        self._remove_data_fields_used_in_condition = remove_data_fields_used_in_condition

    @override
    def _process(self, data: SampleDataGroup) -> SampleDataGroup:
        # Make sure annotations have all needed fields and set output fields
        annotation_paths = data.find_all_occurrences(self._annotation_field_name)
        for ap in annotation_paths:
            annotations = data.get_item_in_path(ap)
            self._eval_and_set_result_for_group(annotations)

        if self._remove_data_fields_used_in_condition:
            self._remove_condition_fields(data)

        return data

    @override
    def _check_and_adjust_data_format_input_to_output(self, data_empty: SampleDataGroup) -> SampleDataGroup:

        annotation_paths = data_empty.find_all_occurrences(self._annotation_field_name)
        used_fields = self._get_needed_annotation_names(self._condition)
        if len(annotation_paths) == 0:
            raise ValueError(
                f"No occurrences of annotations found. Annotation data group fields are expected to have the name '{self._annotation_field_name}', as specified in the constructor."
            )
        for ap in annotation_paths:
            annotation = data_empty.get_item_in_path(ap)
            annotation.check_has_children(data_field_children=used_fields, current_name=str(ap))
            annotation.add_data_field(self._result_field_name, types.DALIDataType.BOOL)

        if self._remove_data_fields_used_in_condition:
            self._remove_condition_fields(data_empty)

        return data_empty

    def _eval_and_set_result_for_group(self, annotations: SampleDataGroup) -> SampleDataGroup:
        valid = self._eval_condition_tree(annotations, self._condition)
        annotations.add_data_field(self._result_field_name, types.DALIDataType.BOOL)
        annotations[self._result_field_name] = valid
        return annotations

    def _remove_condition_fields(self, data_input: SampleDataGroup):
        '''Remove the fields used in the condition from the annotation data group.

        Args:
            data_input: Data to be processed by the step.
        '''
        annotation_paths = data_input.find_all_occurrences(self._annotation_field_name)
        used_fields = self._get_needed_annotation_names(self._condition)
        for ap in annotation_paths:
            annotation = data_input.get_item_in_path(ap)
            for field in used_fields:
                annotation.remove_field(field)

    @staticmethod
    def _eval_condition_tree(annotation: SampleDataGroup, condition: AST):
        if isinstance(condition, Comparison):
            res = AnnotationElementConditionEval._eval_comparison(annotation, condition)
        elif isinstance(condition, Not):
            res = AnnotationElementConditionEval._eval_not(annotation, condition)
        elif isinstance(condition, UnaryMinus):
            res = AnnotationElementConditionEval._eval_unary_minus(annotation, condition)
        elif isinstance(condition, (And, Or)):
            res = AnnotationElementConditionEval._eval_logical_combination(annotation, condition)
        elif isinstance(condition, Variable):
            res = AnnotationElementConditionEval._eval_variable(annotation, condition)
        elif isinstance(condition, Literal):
            res = AnnotationElementConditionEval._eval_literal(condition)
        else:
            raise NotImplementedError(f"Condition type not supported: {type(condition)}")
        return res

    @staticmethod
    def _eval_logical_combination(annotation: SampleDataGroup, combination: Union[Or, And]):
        conditions_to_combine = combination.conditions
        is_and = isinstance(combination, And)
        assert is_and or isinstance(
            combination, Or
        ), "`combination` has to be either be of type `AnnotationElementConditionEval.And` or `AnnotationElementConditionEval.Or`"
        res = True if is_and else False
        for cc in conditions_to_combine:
            res_cc = AnnotationElementConditionEval._eval_condition_tree(annotation, cc)
            # Ensure that the comparison corresponds to the logical AND/OR operation
            # by converting the input to bool. Note that `res` is already a bool and
            # remains so throughout the evaluation.
            res_cc = fn.cast(res_cc, dtype=types.DALIDataType.BOOL)
            if is_and:
                res = res & res_cc
            else:
                res = res | res_cc
        return res

    @staticmethod
    def _eval_not(annotation: SampleDataGroup, combination: Not):
        cond_res = AnnotationElementConditionEval._eval_condition_tree(annotation, combination.condition)
        # Ensure that the operation is logical NOT by converting the input to bool.
        cond_res = fn.cast(cond_res, dtype=types.DALIDataType.BOOL)
        # The `not` operator is only supported for 0D tensors. Therefore, use `!= True` instead.
        res = cond_res != True
        return res

    @staticmethod
    def _eval_unary_minus(annotation: SampleDataGroup, combination: UnaryMinus):
        cond_res = AnnotationElementConditionEval._eval_condition_tree(annotation, combination.value)
        res = -cond_res
        return res

    @staticmethod
    def _eval_comparison(annotation: SampleDataGroup, comparison: Comparison):
        data1 = AnnotationElementConditionEval._eval_condition_tree(annotation, comparison.val1)
        data2 = AnnotationElementConditionEval._eval_condition_tree(annotation, comparison.val2)
        # Perform the requested comparison operation. Note that the if statements are evaluated at DALI graph
        # construction time and only the actual comparison becomes part of the graph & is executed at
        # run time.

        # Note that we need to cast the data to float (a common type) for the comparison to work correctly.
        data1 = fn.cast(data1, dtype=types.DALIDataType.FLOAT)
        data2 = fn.cast(data2, dtype=types.DALIDataType.FLOAT)
        if comparison.comparison_type == "==":
            res = data1 == data2
        elif comparison.comparison_type == "!=":
            res = data1 != data2
        elif comparison.comparison_type == "<":
            res = data1 < data2
        elif comparison.comparison_type == ">":
            res = data1 > data2
        elif comparison.comparison_type == "<=":
            res = data1 <= data2
        elif comparison.comparison_type == ">=":
            res = data1 >= data2
        else:
            raise NotImplementedError(
                f"Comparison operation needs to be one of the following: '==', '!=', '<', '>', '<=', '>='; Got: '{comparison.comparison_type}'"
            )
        return res

    @staticmethod
    def _eval_variable(annotation: SampleDataGroup, variable: Variable):
        data = annotation[variable.name]
        data_shape = data.shape()
        data_num_dims = data_shape.shape()[0]
        if data_num_dims > 1:
            if data_shape[0] == 1:
                data = fn.squeeze(data, axes=[0])
            else:
                data = fn.squeeze(data, axes=[1])
        return data

    @staticmethod
    def _eval_literal(literal: Literal):
        return float(literal.value)

    @staticmethod
    def _get_needed_annotation_names(condition: AST):
        # collect all used data field names (will contain duplicates if fields are used multiple times)
        res_with_repetition = AnnotationElementConditionEval._get_needed_annotation_names_rec(condition)
        # remove duplicates from the used data field names
        res = list(dict.fromkeys(res_with_repetition))
        return res

    @staticmethod
    def _get_needed_annotation_names_rec(condition: AST):
        if isinstance(condition, Variable):
            return [condition.name]
        elif isinstance(condition, (And, Or)):
            res = []
            for c in condition.conditions:
                res = res + AnnotationElementConditionEval._get_needed_annotation_names_rec(c)
            return res
        elif isinstance(condition, Not):
            return AnnotationElementConditionEval._get_needed_annotation_names_rec(condition.condition)
        elif isinstance(condition, Comparison):
            return AnnotationElementConditionEval._get_needed_annotation_names_rec(
                condition.val1
            ) + AnnotationElementConditionEval._get_needed_annotation_names_rec(condition.val2)
        else:
            return []
