"""词法分析器模块。

将因子表达式字符串拆分为 Token 序列，支持：
- 变量（$VAR_NAME）
- 数字（整数、浮点、科学计数法）
- 算术运算符（+ - * / %）
- 比较运算符（> < >= <= == !=）
- 逻辑运算符（&& || !）
- 条件表达式关键字（IF THEN ELSE）
- 函数调用（NAME(args...)）
- 括号、逗号等分隔符
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class TokenType(Enum):
    """词法单元类型枚举。"""

    # 字面量
    NUMBER = auto()       # 数字（含科学计数法）
    STRING = auto()       # 字符串字面量
    VAR = auto()          # 变量引用 ($VAR)
    IDENTIFIER = auto()   # 标识符（函数名等）

    # 算术运算符
    PLUS = auto()         # +
    MINUS = auto()        # -
    STAR = auto()         # *
    SLASH = auto()        # /
    PERCENT = auto()      # %
    CARET = auto()        # ^

    # 比较运算符
    GT = auto()           # >
    GTE = auto()          # >=
    LT = auto()           # <
    LTE = auto()          # <=
    EQ = auto()           # ==
    NEQ = auto()          # !=

    # 逻辑运算符
    AND = auto()          # &&
    OR = auto()           # ||
    NOT = auto()          # !

    # 关键字
    IF = auto()           # IF
    ELSE = auto()         # ELSE
    
    # 条件表达式
    QUESTION = auto()     # ?
    COLON = auto()        # :

    # 分隔符
    LPAREN = auto()       # (
    RPAREN = auto()       # )
    COMMA = auto()        # ,
    
    # 特殊
    EOF = auto()          # 结束
    ILLEGAL = auto()      # 非法字符


@dataclass
class Token:
    """词法单元。"""

    type: TokenType
    value: str
    position: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, pos={self.position})"


# 关键字映射
_KEYWORDS: dict[str, TokenType] = {
    "IF": TokenType.IF,
    "if": TokenType.IF,
    "ELSE": TokenType.ELSE,
    "else": TokenType.ELSE,
    "TRUE": TokenType.NUMBER,
    "FALSE": TokenType.NUMBER,
}


class ExpressionLexer:
    """因子表达式词法分析器。

    将输入字符串转换为 Token 序列，供 Parser 使用。
    支持两种使用方式:
    1. 无状态: lexer.tokenize("$close > $open")
    2. 有状态: lexer = ExpressionLexer("$close > $open"); lexer.tokenize()
    """

    def __init__(self, text: Optional[str] = None) -> None:
        """初始化词法分析器。

        Args:
            text: 待分析的表达式字符串（可选）。
        """
        self._text: str = text or ""
        self._pos: int = 0
        self._tokens: List[Token] = []

    def tokenize(self, text: Optional[str] = None) -> List[Token]:
        """将表达式拆分为 Token 列表。

        Args:
            text: 待分析的表达式字符串（可选，覆盖构造函数传入的值）。

        Returns:
            Token 列表（末尾包含 EOF Token）。
        """
        if text is not None:
            self._text = text
        self._tokens = []
        self._pos = 0

        while self._pos < len(self._text):
            self._skip_whitespace()
            if self._pos >= len(self._text):
                break

            token = self._next_token()
            if token is not None:
                self._tokens.append(token)

        self._tokens.append(Token(TokenType.EOF, "", self._pos))
        return self._tokens

    def _skip_whitespace(self) -> None:
        """跳过空白字符。"""
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _next_token(self) -> Optional[Token]:
        """从当前位置读取下一个 Token。

        Returns:
            下一个 Token，若为注释则跳过返回 None。
        """
        pos = self._pos
        ch = self._text[pos]

        # 变量引用
        if ch == "$":
            return self._read_var()

        # 数字（含科学计数法）
        if ch.isdigit() or (ch == "." and pos + 1 < len(self._text) and self._text[pos + 1].isdigit()):
            return self._read_number()

        # 字符串字面量
        if ch in ('"', "'"):
            return self._read_string()

        # 标识符 / 关键字
        if ch.isalpha() or ch == "_":
            return self._read_identifier()

        # 双字符运算符
        two_char = self._text[pos : pos + 2]
        if two_char == ">=":
            self._pos += 2
            return Token(TokenType.GTE, ">=", pos)
        if two_char == "<=":
            self._pos += 2
            return Token(TokenType.LTE, "<=", pos)
        if two_char == "==":
            self._pos += 2
            return Token(TokenType.EQ, "==", pos)
        if two_char == "!=":
            self._pos += 2
            return Token(TokenType.NEQ, "!=", pos)
        if two_char == "&&":
            self._pos += 2
            return Token(TokenType.AND, "&&", pos)
        if two_char == "||":
            self._pos += 2
            return Token(TokenType.OR, "||", pos)

        # 单字符运算符 / 分隔符
        single_map: dict[str, TokenType] = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "^": TokenType.CARET,
            ">": TokenType.GT,
            "<": TokenType.LT,
            "!": TokenType.NOT,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            ",": TokenType.COMMA,
            "?": TokenType.QUESTION,
            ":": TokenType.COLON,
        }

        if ch in single_map:
            self._pos += 1
            return Token(single_map[ch], ch, pos)

        # 非法字符
        self._pos += 1
        return Token(TokenType.ILLEGAL, ch, pos)

    def _read_var(self) -> Token:
        """读取变量引用（$VAR_NAME）。"""
        start = self._pos
        self._pos += 1  # 跳过 $

        if self._pos >= len(self._text) or not (
            self._text[self._pos].isalpha() or self._text[self._pos] == "_"
        ):
            return Token(TokenType.ILLEGAL, "$", start)

        while self._pos < len(self._text) and (
            self._text[self._pos].isalnum() or self._text[self._pos] == "_"
        ):
            self._pos += 1

        value = self._text[start:self._pos]
        return Token(TokenType.VAR, value, start)

    def _read_number(self) -> Token:
        """读取数字（整数、浮点、科学计数法）。"""
        start = self._pos
        has_dot = False
        has_exp = False

        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch.isdigit():
                self._pos += 1
            elif ch == "." and not has_dot:
                has_dot = True
                self._pos += 1
            elif ch in ("e", "E") and not has_exp:
                has_exp = True
                self._pos += 1
                # 科学计数法指数部分的正负号
                if self._pos < len(self._text) and self._text[self._pos] in ("+", "-"):
                    self._pos += 1
            else:
                break

        value = self._text[start:self._pos]
        return Token(TokenType.NUMBER, value, start)

    def _read_string(self) -> Token:
        """读取字符串字面量。"""
        start = self._pos
        quote = self._text[self._pos]
        self._pos += 1

        while self._pos < len(self._text) and self._text[self._pos] != quote:
            if self._text[self._pos] == "\\":
                self._pos += 2  # 跳过转义
            else:
                self._pos += 1

        if self._pos < len(self._text):
            self._pos += 1  # 跳过结束引号

        value = self._text[start:self._pos]
        return Token(TokenType.STRING, value, start)

    def _read_identifier(self) -> Token:
        """读取标识符或关键字。"""
        start = self._pos

        while self._pos < len(self._text) and (
            self._text[self._pos].isalnum() or self._text[self._pos] == "_"
        ):
            self._pos += 1

        value = self._text[start:self._pos]
        token_type = _KEYWORDS.get(value, TokenType.IDENTIFIER)

        # TRUE/FALSE 作为 NUMBER (0.0 / 1.0)
        if value.upper() == "TRUE":
            return Token(TokenType.NUMBER, "1", start)
        if value.upper() == "FALSE":
            return Token(TokenType.NUMBER, "0", start)

        return Token(token_type, value, start)
