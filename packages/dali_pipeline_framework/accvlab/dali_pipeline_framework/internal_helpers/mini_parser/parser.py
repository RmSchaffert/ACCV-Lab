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


from .lexer import Token, TokenType, Lexer
from . import ast


class Parser:
    '''The actual parser.

    The input string must start with a variable name, followed by an assignment operator, followed by an
    expression.

    The expression is parsed recursively. The expression can contain variables, literals, and operators.

    The operators are:
        - Logical operators: ``or``, ``and``, ``not``
        - Comparison operators: ``==``, ``!=``, ``>``, ``>=``, ``<``, ``<=``
        - Parentheses: ``(`` and ``)``
        - Unary minus: ``-``
        - Assignment operator: ``=``

    The syntax is similar to Python. However, note that
        - Only the operators defined above are supported.
        - Comparisons of more than two values are not supported (e.g. ``a < b < c`` is not supported).
        - Only numeric literals are supported. ``True`` and ``False`` are not supported (not needed in the current
          use case).

    Example:

        Some valid statements:
          - ``res_var = -_b1 < 10.5``
          - ``res_1_var = -_b1 < 10.5 and -c > -20``
          - ``res_3_var = not -_b1 < 10.5``
          - ``res_4_var = (-_b1 < 10.5 or a_bool_var) and another_nool_var``
          - ``res_5_var = (-_b1 < 10.5 or (-c > -20 and d == 10)) and another_var > 30``
          - ``res_7_var = (-_b1 < 10.5 or (-c > -20 and d == 10)) and (another_var > 30 and _e < 40) and f > 50``


    Args:
        input_str: Input string to parse.
    '''

    _priority_map = {
        TokenType.LOGICAL_OR: 1,
        TokenType.LOGICAL_AND: 2,
        TokenType.COMPARISON: 3,
        TokenType.LOGICAL_NOT: 4,
        TokenType.MINUS: 4,
        TokenType.PARENTHESIS_OPEN: 5,
    }

    def __init__(self, input_str: str):
        self._lexer = Lexer(input_str)
        self._tokens = []
        while True:
            token = self._lexer.next_token()
            self._tokens.append(token)
            if token.type == TokenType.EOL:
                break
        self._curr_token_idx = 0

    def parse(self) -> ast.AST:
        '''Parses the input stream and returns an AST.

        See the class docstring for the syntax.

        Returns:
            The AST of the input stream.
        '''
        var = self._curr_token()
        if var.type != TokenType.VARIABLE:
            raise ValueError(
                f"The condition must start with `<res_var_name> = ...` (replace `<res_var_name>` with the name of the variable to store the result in)"
            )
        self._move_to_next_token()
        assignment = self._curr_token()
        if assignment.type != TokenType.ASSIGNMENT:
            raise ValueError(
                f"The condition must start with `<res_var_name> = ...` (replace `<res_var_name>` with the name of the variable to store the result in)"
            )
        self._move_to_next_token()
        expression = self._parse_expression(0)
        res = ast.Assignment(ast.Variable(var.value), expression)
        return res

    def _curr_token(self) -> Token:
        '''Returns the current token.

        Returns:
            The current token.
        '''
        return self._tokens[self._curr_token_idx]

    def _curr_token_priority(self) -> int:
        '''Returns the priority of the current token.

        Returns:
            The priority of the current token.
        '''
        return self._priority_map[self._curr_token().type]

    def _move_to_next_token(self):
        '''Move current token forward'''
        self._curr_token_idx += 1

    def _parse_expression(self, priority: int) -> ast.AST:
        '''Parses the input stream and returns an AST.

        Args:
            priority: The priority of the expression to parse.
        '''
        left = self._parse_prefix()
        end_token_types = (TokenType.EOL, TokenType.PARENTHESIS_CLOSE)
        while not self._curr_token().type in end_token_types and self._curr_token_priority() > priority:
            left = self._parse_infix(left, self._curr_token_priority())
        return left

    def _parse_prefix(self) -> ast.AST:
        curr_token = self._curr_token()
        if curr_token.type in (TokenType.LOGICAL_NOT, TokenType.MINUS):
            priority = self._curr_token_priority()
            self._move_to_next_token()
            inner = self._parse_expression(priority)
            if curr_token.type == TokenType.LOGICAL_NOT:
                res = ast.Not(inner)
            elif curr_token.type == TokenType.MINUS:
                res = ast.UnaryMinus(inner)
            return res
        elif curr_token.type == TokenType.PARENTHESIS_OPEN:
            self._move_to_next_token()
            # Parse the inner expression.
            # We reset the priority to 0 because we are parsing the inner expression
            res = self._parse_expression(0)
            # The right parenthesis serves as the end token for the inner expression.
            # We do not advance it inside the inner expression (as it may need to serve as the end token
            # for multiple recursion levels). Instead, we advance it here.
            self._move_to_next_token()
            return res
        elif curr_token.type == TokenType.LITERAL or curr_token.type == TokenType.VARIABLE:
            if curr_token.type == TokenType.LITERAL:
                value = ast.Literal(curr_token.value)
            elif curr_token.type == TokenType.VARIABLE:
                value = ast.Variable(curr_token.value)
            self._move_to_next_token()
            return value
        else:
            raise ValueError(f"Invalid token: {curr_token}")

    def _parse_infix(self, left_value: ast.AST, priority: int) -> ast.AST:
        '''Parses an infix expression from the input stream and returns an AST.

        Args:
            left_value: The left value of the infix expression.
            priority: The priority of the infix expression.
        '''
        combination = self._curr_token()
        self._move_to_next_token()
        right_value = self._parse_expression(priority)
        if combination.type == TokenType.LOGICAL_OR:
            res = ast.Or(left_value, right_value)
        elif combination.type == TokenType.LOGICAL_AND:
            res = ast.And(left_value, right_value)
        elif combination.type == TokenType.COMPARISON:
            res = ast.Comparison(left_value, combination.value, right_value)
        return res
