"""LL(1) 递归下降解析器模块。

将 Token 序列解析为 AST。支持完整的运算符优先级、
函数调用、条件表达式及嵌套括号。

运算符优先级（由低到高）：
    ||  →  &&  →  == !=  →  > < >= <=  →  + -  →  * / %  →  ^  →  一元(- !)  →  主要表达式
"""

from __future__ import annotations

from typing import List, Optional

from .lexer import ExpressionLexer, Token, TokenType
from .ast import (
    ASTNode,
    BinaryOpNode,
    ConditionalNode,
    FuncCallNode,
    NumNode,
    StringNode,
    UnaryOpNode,
    VarNode,
)


class ParseError(Exception):
    """解析错误异常。"""

    def __init__(self, message: str, token: Optional[Token] = None) -> None:
        self.token = token
        if token is not None:
            message = f"{message} (at position {token.position})"
        super().__init__(message)


class ExpressionParser:
    """LL(1) 递归下降解析器。

    使用 Pratt 解析（Precedence Climbing）处理二元运算符优先级，
    主要表达式采用递归下降。
    支持两种使用方式:
    1. 无状态: parser.parse(tokens)
    2. 有状态: parser = ExpressionParser(tokens); parser.parse()
    """

    def __init__(self, tokens: Optional[List[Token]] = None) -> None:
        """初始化解析器。

        Args:
            tokens: 由 ExpressionLexer 生成的 Token 列表（可选）。
        """
        self._tokens: List[Token] = tokens or []
        self._pos: int = 0

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def parse(self, tokens: Optional[List[Token]] = None) -> ASTNode:
        """解析整个表达式并返回 AST 根节点。

        Args:
            tokens: Token 列表（可选，覆盖构造函数传入的值）。

        Returns:
            解析后的 AST 节点。

        Raises:
            ParseError: 解析出错时抛出。
        """
        if tokens is not None:
            self._tokens = tokens
        self._pos = 0
        node = self._parse_expression()
        if not self._is_at_end():
            raise ParseError(
                f"意外的 token: {self._peek().type.name}", self._peek()
            )
        return node

    # ------------------------------------------------------------------
    # Token 流操作
    # ------------------------------------------------------------------

    def _peek(self, offset: int = 0) -> Token:
        """查看当前位置的 Token（不消费）。"""
        idx = self._pos + offset
        if idx < len(self._tokens):
            return self._tokens[idx]
        return self._tokens[-1]  # EOF

    def _advance(self) -> Token:
        """消费当前 Token 并返回。"""
        token = self._tokens[self._pos]
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return token

    def _is_at_end(self) -> bool:
        """是否到达 Token 流末尾。"""
        return self._peek().type == TokenType.EOF

    def _check(self, *types: TokenType) -> bool:
        """检查当前 Token 是否属于给定类型。"""
        return self._peek().type in types

    def _match(self, *types: TokenType) -> bool:
        """若当前 Token 匹配则消费并返回 True。"""
        if self._check(*types):
            self._advance()
            return True
        return False

    def _expect(self, token_type: TokenType) -> Token:
        """期望当前 Token 为指定类型，否则抛错。"""
        token = self._peek()
        if token.type != token_type:
            raise ParseError(
                f"期望 {token_type.name}，实际为 {token.type.name}", token
            )
        return self._advance()

    # ------------------------------------------------------------------
    # 优先级解析（Precedence Climbing）
    # ------------------------------------------------------------------

    # 运算符优先级表
    _PRECEDENCE: dict[TokenType, int] = {
        TokenType.OR: 1,
        TokenType.AND: 2,
        TokenType.EQ: 3,
        TokenType.NEQ: 3,
        TokenType.GT: 4,
        TokenType.GTE: 4,
        TokenType.LT: 4,
        TokenType.LTE: 4,
        TokenType.PLUS: 5,
        TokenType.MINUS: 5,
        TokenType.STAR: 6,
        TokenType.SLASH: 6,
        TokenType.PERCENT: 6,
        TokenType.CARET: 7,
    }

    def _parse_expression(self) -> ASTNode:
        """解析表达式入口点（支持条件表达式）。"""
        return self._parse_conditional()
    
    def _parse_conditional(self) -> ASTNode:
        """解析条件表达式 (condition ? true_expr : false_expr)。"""
        condition = self._parse_or()
        
        if self._match(TokenType.QUESTION):
            true_expr = self._parse_expression()
            self._expect(TokenType.COLON)
            false_expr = self._parse_expression()
            return ConditionalNode(condition, true_expr, false_expr)
        
        return condition

    def _parse_or(self) -> ASTNode:
        """解析 OR 表达式（最低优先级）。"""
        left = self._parse_and()

        while self._match(TokenType.OR):
            right = self._parse_and()
            left = BinaryOpNode("||", left, right)

        return left

    def _parse_and(self) -> ASTNode:
        """解析 AND 表达式。"""
        left = self._parse_equality()

        while self._match(TokenType.AND):
            right = self._parse_equality()
            left = BinaryOpNode("&&", left, right)

        return left

    def _parse_equality(self) -> ASTNode:
        """解析相等性表达式。"""
        left = self._parse_comparison()

        while True:
            if self._match(TokenType.EQ):
                right = self._parse_comparison()
                left = BinaryOpNode("==", left, right)
            elif self._match(TokenType.NEQ):
                right = self._parse_comparison()
                left = BinaryOpNode("!=", left, right)
            else:
                break

        return left

    def _parse_comparison(self) -> ASTNode:
        """解析比较表达式。"""
        left = self._parse_addition()

        while True:
            if self._match(TokenType.GT):
                right = self._parse_addition()
                left = BinaryOpNode(">", left, right)
            elif self._match(TokenType.GTE):
                right = self._parse_addition()
                left = BinaryOpNode(">=", left, right)
            elif self._match(TokenType.LT):
                right = self._parse_addition()
                left = BinaryOpNode("<", left, right)
            elif self._match(TokenType.LTE):
                right = self._parse_addition()
                left = BinaryOpNode("<=", left, right)
            else:
                break

        return left

    def _parse_addition(self) -> ASTNode:
        """解析加减表达式。"""
        left = self._parse_multiplication()

        while True:
            if self._match(TokenType.PLUS):
                right = self._parse_multiplication()
                left = BinaryOpNode("+", left, right)
            elif self._match(TokenType.MINUS):
                right = self._parse_multiplication()
                left = BinaryOpNode("-", left, right)
            else:
                break

        return left

    def _parse_multiplication(self) -> ASTNode:
        """解析乘除取模表达式。"""
        left = self._parse_power()

        while True:
            if self._match(TokenType.STAR):
                right = self._parse_power()
                left = BinaryOpNode("*", left, right)
            elif self._match(TokenType.SLASH):
                right = self._parse_power()
                left = BinaryOpNode("/", left, right)
            elif self._match(TokenType.PERCENT):
                right = self._parse_power()
                left = BinaryOpNode("%", left, right)
            else:
                break

        return left

    def _parse_power(self) -> ASTNode:
        """解析幂运算表达式（右结合）。"""
        base = self._parse_unary()

        if self._match(TokenType.CARET):
            exponent = self._parse_power()  # 右结合
            return BinaryOpNode("^", base, exponent)

        return base

    def _parse_unary(self) -> ASTNode:
        """解析一元运算。"""
        if self._match(TokenType.MINUS):
            operand = self._parse_unary()
            return UnaryOpNode("-", operand)

        if self._match(TokenType.NOT):
            operand = self._parse_unary()
            return UnaryOpNode("!", operand)

        return self._parse_postfix()

    # ------------------------------------------------------------------
    # 主要表达式
    # ------------------------------------------------------------------

    def _parse_postfix(self) -> ASTNode:
        """解析后缀表达式（目前仅用于扩展）。"""
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        """解析主要表达式。

        处理：数字、字符串、变量、函数调用、条件表达式、括号表达式。
        """
        token = self._peek()

        # 数字
        if token.type == TokenType.NUMBER:
            self._advance()
            return NumNode(value=float(token.value))

        # 字符串
        if token.type == TokenType.STRING:
            self._advance()
            # 去除引号
            value = token.value
            if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
                value = value[1:-1]
            return StringNode(value=value)

        # 变量引用
        if token.type == TokenType.VAR:
            self._advance()
            return VarNode(name=token.value)

        # 条件表达式 IF(cond, then, else)
        if token.type == TokenType.IF:
            return self._parse_if_expression()

        # 标识符（可能是函数调用）
        if token.type == TokenType.IDENTIFIER:
            # 向前查看是否为 LPAREN
            if self._peek(1).type == TokenType.LPAREN:
                return self._parse_function_call()
            # 独立标识符视为变量引用
            self._advance()
            return VarNode(name=f"${token.value}")

        # 括号表达式
        if token.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return expr

        # 错误
        raise ParseError(
            f"意外的 token: {token.type.name} ({token.value!r})", token
        )

    def _parse_if_expression(self) -> ASTNode:
        """解析条件表达式 IF(cond, then, else)。"""
        self._expect(TokenType.IF)
        self._expect(TokenType.LPAREN)

        condition = self._parse_expression()
        self._expect(TokenType.COMMA)
        then_branch = self._parse_expression()

        else_branch: Optional[ASTNode] = None
        if self._match(TokenType.COMMA):
            else_branch = self._parse_expression()

        self._expect(TokenType.RPAREN)
        return ConditionalNode(
            condition=condition,
            then_branch=then_branch,
            else_branch=else_branch,
        )

    def _parse_function_call(self) -> ASTNode:
        """解析函数调用 NAME(args...)。"""
        name_token = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.LPAREN)

        args: List[ASTNode] = []
        if not self._check(TokenType.RPAREN):
            args.append(self._parse_expression())
            while self._match(TokenType.COMMA):
                args.append(self._parse_expression())

        self._expect(TokenType.RPAREN)
        return FuncCallNode(name=name_token.value, args=args)
