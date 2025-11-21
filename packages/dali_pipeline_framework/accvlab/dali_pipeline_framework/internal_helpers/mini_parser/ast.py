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


class AST:
    '''Base class for the AST nodes.'''

    pass


class Assignment(AST):
    '''Represents an assignment of a value to a variable.

    Note:
        The constructor parameters are also the attributes of the class.

    Args:
        variable: Variable to assign the value to.
        expression: Expression to evaluate and assign the result to the variable.
    '''

    def __init__(self, variable: 'Variable', expression: AST):
        self.variable = variable
        self.expression = expression

    def __str__(self):
        return f"{self.variable} = <{self.expression}>"

    def __repr__(self):
        return self.__str__()


class Literal(AST):
    '''Represents a literal value.

    Note:
        The constructor parameters are also the attributes of the class.

    Args:
        value: Value to represent.
    '''

    def __init__(self, value: str):
        self.value = value

    def __str__(self):
        return f"{self.value}"

    def __repr__(self):
        return self.__str__()


class Variable(AST):
    '''Represents a variable (by name).

    Note:
        The constructor parameters are also the attributes of the class.

    Args:
        name: Name of the variable.
    '''

    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"'{self.name}'"

    def __repr__(self):
        return self.__str__()


class Comparison(AST):
    '''Represents a comparison.

    Note:
        The constructor parameters are also the attributes of the class.

    Args:
        val1: First value to compare.
        comparison_type: Type of comparison to perform. Supported comparisons
            are ``==``, ``!=``, ``<``, ``>``, ``>=``, ``<=``.
        val2: Second value to compare.
    '''

    def __init__(self, val1: Variable, comparison_type: str, val2: Literal):
        allowed_comparison_types = ["==", "!=", "<", ">", ">=", "<="]
        if not comparison_type in allowed_comparison_types:
            raise ValueError(
                f"`comparison_type` has to be one of the following options: {allowed_comparison_types}. Got \"{comparison_type}\" instead"
            )
        self.val1 = val1
        self.comparison_type = comparison_type
        self.val2 = val2

    def __str__(self):
        return f"({self.val1} {self.comparison_type} {self.val2})"

    def __repr__(self):
        return self.__str__()


class Or(AST):
    '''Represents a logical ``or`` on a set of operands.

    Note:
        The constructor parameters are also the attributes of the class.
        The individual conditions are stored as a tuple.

    Args:
        conditions: Conditions to be combined with ``or``. Note that the number of conditions is variable,
            i.e. more than two conditions can be combined.
    '''

    def __init__(self, *conditions: AST):
        self.conditions = conditions

    def __str__(self):
        return f"{{{' or '.join(f'{cond}' for cond in self.conditions)}}}"

    def __repr__(self):
        return self.__str__()


class And(AST):
    '''Represents a logical ``and`` on a set of operands.

    Note:
        The constructor parameters are also the attributes of the class.
        The individual conditions are stored as a tuple.

    Args:
        conditions: Conditions to be combined with ``and``. Note that the number of conditions is variable,
            i.e. more than two conditions can be combined.
    '''

    def __init__(self, *conditions: AST):
        self.conditions = conditions

    def __str__(self):
        return f"{{{' and '.join(f'{cond}' for cond in self.conditions)}}}"

    def __repr__(self):
        return self.__str__()


class Not(AST):
    '''Represents a logical ``not`` (i.e. negation) for an operand.

    Note:
        The constructor parameters are also the attributes of the class.

    Args:
        condition: Condition to be negated.
    '''

    def __init__(self, condition: AST):
        self.condition = condition

    def __str__(self):
        return f"not [{self.condition}]"

    def __repr__(self):
        return self.__str__()


class UnaryMinus(AST):
    '''Represents a unary minus.

    Note:
        The constructor parameters are also the attributes of the class.
    '''

    def __init__(self, value: AST):
        self.value = value

    def __str__(self):
        return f"-({self.value})"

    def __repr__(self):
        return self.__str__()
