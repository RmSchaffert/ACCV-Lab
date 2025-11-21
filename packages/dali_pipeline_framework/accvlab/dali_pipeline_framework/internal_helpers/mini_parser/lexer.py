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


class TokenType:
    '''Represents the type of a token'''

    LITERAL = "literal"
    VARIABLE = "variable"
    ASSIGNMENT = "assignment"
    COMPARISON = "comparison"
    LOGICAL_OR = "logical_or"
    LOGICAL_AND = "logical_and"
    LOGICAL_NOT = "logical_not"
    MINUS = "minus"
    PARENTHESIS_OPEN = "parenthesis_open"
    PARENTHESIS_CLOSE = "parenthesis_close"
    EOL = "end_of_line"


class Token:
    '''Represents a token in the input stream.

    Note:
        The constructor parameters become attributes of the class.

    Args:
        type: Type of the token.
        value: Value of the token.
    '''

    def __init__(self, type: TokenType, value: str):
        self.type = type
        self.value = value

    def __str__(self):
        return f"Token(type='{self.type}', value='{self.value}')"

    def __repr__(self):
        return self.__str__()


class Lexer:
    '''Lexer for the simple parser.

    Args:
        input: Input string to generate tokens from.
    '''

    _keyword_map = {
        "or": TokenType.LOGICAL_OR,
        "and": TokenType.LOGICAL_AND,
        "not": TokenType.LOGICAL_NOT,
    }
    _comparison_map_2_char = {
        "==": TokenType.COMPARISON,
        "!=": TokenType.COMPARISON,
        ">=": TokenType.COMPARISON,
        "<=": TokenType.COMPARISON,
    }
    _comparison_map_1_char = {
        ">": TokenType.COMPARISON,
        "<": TokenType.COMPARISON,
    }
    _comparison_start_chars = {">", "<", "=", "!"}
    _assignment = {
        "=": TokenType.ASSIGNMENT,
    }

    _minus_map = {
        "-": TokenType.MINUS,
    }

    def __init__(self, input: str):
        self._input = input
        self._position = 0

    def next_token(self) -> Token:
        '''Returns the next token in the input stream'''
        if self._position >= len(self._input):
            return Token(TokenType.EOL, "")
        self._skip_whitespaces()
        current_char = self._input[self._position]
        if self._is_digit_dot(current_char):
            return self._process_number()
        elif self._is_alpha_underscore(current_char):
            return self._process_string()
        else:
            return self._process_char()

    def _is_alpha_underscore(self, char: str) -> bool:
        '''Checks if a character is an alpha character or an underscore'''
        return char.isalpha() or char == "_"

    def _is_alpha_underscore_digit(self, char: str) -> bool:
        '''Checks if a character is an alpha character, an underscore, or a digit'''
        return self._is_alpha_underscore(char) or char.isdigit()

    def _is_digit_dot(self, char: str) -> bool:
        '''Checks if a character is a digit or a dot'''
        return char.isdigit() or char == "."

    def _skip_whitespaces(self):
        '''Skips whitespace in the input stream'''
        while self._position < len(self._input) and self._input[self._position].isspace():
            self._position += 1

    def _process_number(self) -> Token:
        '''Parses a number from the input stream'''
        start_pos = self._position
        while self._position < len(self._input) and self._is_digit_dot(self._input[self._position]):
            self._position += 1
        res_content = self._input[start_pos : self._position]
        if res_content.count(".") > 1:
            raise ValueError(f"Invalid number: {res_content}")
        return Token(TokenType.LITERAL, res_content)

    def _process_string(self) -> Token:
        '''Parses a string from the input stream'''

        def check_type(input: str) -> TokenType:
            if input in self._keyword_map:
                return self._keyword_map[input]
            return TokenType.VARIABLE

        if not self._is_alpha_underscore(self._input[self._position]):
            raise ValueError(
                f"Invalid identifier starting at position {self._position}: {self._input[self._position]}"
            )

        start_pos = self._position
        while self._position < len(self._input) and self._is_alpha_underscore_digit(
            self._input[self._position]
        ):
            self._position += 1
        token_str = self._input[start_pos : self._position]
        return Token(check_type(token_str), token_str)

    def _process_char(self) -> Token:
        '''Parses a single character from the input stream'''
        if self._input[self._position] in self._minus_map:
            return self._process_minus()
        if self._input[self._position] in self._comparison_start_chars:
            return self._process_assignment_and_comparison()
        else:
            return self._process_parenthesis()
            raise ValueError(f"Invalid character: {self._input[self._position]}")

    def _process_minus(self) -> Token:
        '''Parses a minus from the input stream'''
        self._position += 1
        return Token(TokenType.MINUS, "-")

    def _process_assignment_and_comparison(self) -> Token:
        '''Parses a comparison from the input stream'''
        if self._input[self._position : self._position + 2] in self._comparison_map_2_char:
            token_str = self._input[self._position : self._position + 2]
            self._position += 2
            return Token(self._comparison_map_2_char[token_str], token_str)
        elif self._input[self._position] in self._comparison_map_1_char:
            token_str = self._input[self._position]
            self._position += 1
            return Token(self._comparison_map_1_char[token_str], token_str)
        elif self._input[self._position] in self._assignment:
            token_str = self._input[self._position]
            self._position += 1
            return Token(self._assignment[token_str], token_str)
        else:
            raise ValueError(f"Invalid comparison operator: {self._input[self._position]}")

    def _process_parenthesis(self) -> Token:
        '''Parses a parenthesis from the input stream'''
        if self._input[self._position] == "(":
            self._position += 1
            return Token(TokenType.PARENTHESIS_OPEN, "(")
        elif self._input[self._position] == ")":
            self._position += 1
            return Token(TokenType.PARENTHESIS_CLOSE, ")")
        else:
            raise ValueError(f"Invalid parenthesis: {self._input[self._position]}")
