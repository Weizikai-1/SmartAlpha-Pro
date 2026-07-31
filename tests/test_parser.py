"""语法分析器 ExpressionParser 测试套件。

覆盖:
- 简单表达式 (变量、数字)
- 二元运算优先级
- 函数调用 (嵌套、多参数)
- 嵌套括号
- 条件表达式 (三元、IF 函数)
- 错误处理
"""

import pytest

from smartalpha.core.lexer import ExpressionLexer, Token, TokenType
from smartalpha.core.parser import ExpressionParser, ParseError
from smartalpha.core.ast import (
    ASTNode,
    VarNode,
    NumNode,
    StringNode,
    BinaryOpNode,
    UnaryOpNode,
    FuncCallNode,
    ConditionalNode,
)


@pytest.fixture
def parse_expr(lexer, parser):
    """便捷方法: 直接解析表达式字符串。"""
    def _parse(expr: str) -> ASTNode:
        tokens = lexer.tokenize(expr)
        return parser.parse(tokens)
    return _parse


class TestSimpleExpressions:
    """简单表达式测试。"""

    def test_parse_variable(self, parse_expr):
        """解析变量。"""
        ast = parse_expr("$close")
        assert isinstance(ast, VarNode)
        assert ast.name == "$close"

    def test_parse_number_integer(self, parse_expr):
        """解析整数。"""
        ast = parse_expr("42")
        assert isinstance(ast, NumNode)
        assert ast.value == 42.0

    def test_parse_number_float(self, parse_expr):
        """解析浮点数。"""
        ast = parse_expr("3.14")
        assert isinstance(ast, NumNode)
        assert abs(ast.value - 3.14) < 1e-10

    def test_parse_scientific_number(self, parse_expr):
        """解析科学计数法。"""
        ast = parse_expr("1e5")
        assert isinstance(ast, NumNode)
        assert ast.value == 1e5

    def test_parse_string(self, parse_expr):
        """解析字符串字面量。"""
        ast = parse_expr('"hello"')
        assert isinstance(ast, StringNode)
        assert ast.value == "hello"

    def test_parse_single_identifier(self, parse_expr):
        """独立标识符解析为 VarNode (加 $ 前缀)。"""
        ast = parse_expr("close")
        assert isinstance(ast, VarNode)
        assert ast.name == "$close"

    def test_parse_true(self, parse_expr):
        """TRUE 解析为数字 1。"""
        ast = parse_expr("TRUE")
        assert isinstance(ast, NumNode)
        assert ast.value == 1.0

    def test_parse_false(self, parse_expr):
        """FALSE 解析为数字 0。"""
        ast = parse_expr("FALSE")
        assert isinstance(ast, NumNode)
        assert ast.value == 0.0


class TestBinaryOperations:
    """二元运算测试。"""

    def test_addition(self, parse_expr):
        """加法。"""
        ast = parse_expr("$close + $open")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "+"
        assert isinstance(ast.left, VarNode)
        assert isinstance(ast.right, VarNode)

    def test_subtraction(self, parse_expr):
        """减法。"""
        ast = parse_expr("$close - $open")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "-"

    def test_multiplication(self, parse_expr):
        """乘法。"""
        ast = parse_expr("$close * 2")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "*"

    def test_division(self, parse_expr):
        """除法。"""
        ast = parse_expr("$close / $open")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "/"

    def test_modulo(self, parse_expr):
        """取模。"""
        ast = parse_expr("$close % 3")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "%"

    def test_power(self, parse_expr):
        """幂运算。"""
        ast = parse_expr("$close ^ 2")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "^"

    def test_comparison_gt(self, parse_expr):
        """大于比较。"""
        ast = parse_expr("$close > $open")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == ">"

    def test_comparison_gte(self, parse_expr):
        """大于等于比较。"""
        ast = parse_expr("$close >= 100")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == ">="

    def test_comparison_lt(self, parse_expr):
        """小于比较。"""
        ast = parse_expr("$close < $high")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "<"

    def test_comparison_lte(self, parse_expr):
        """小于等于比较。"""
        ast = parse_expr("$close <= 200")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "<="

    def test_comparison_eq(self, parse_expr):
        """等于比较。"""
        ast = parse_expr("$close == $open")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "=="

    def test_comparison_neq(self, parse_expr):
        """不等于比较。"""
        ast = parse_expr("$close != $open")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "!="

    def test_logical_and(self, parse_expr):
        """逻辑与。"""
        ast = parse_expr("$close > 100 && $volume > 1000")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "&&"

    def test_logical_or(self, parse_expr):
        """逻辑或。"""
        ast = parse_expr("$close > 100 || $high < 200")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "||"


class TestOperatorPrecedence:
    """运算符优先级测试。"""

    def test_multiplication_before_addition(self, parse_expr):
        """先乘后加: $close + $high * $low。"""
        ast = parse_expr("$close + $high * $low")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "+"
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.op == "*"

    def test_division_before_subtraction(self, parse_expr):
        """先除后减: $close - $high / $low。"""
        ast = parse_expr("$close - $high / $low")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "-"
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.op == "/"

    def test_power_before_multiplication(self, parse_expr):
        """先幂后乘: 2 * $close ^ 2。"""
        ast = parse_expr("2 * $close ^ 2")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "*"
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.op == "^"

    def test_right_associative_power(self, parse_expr):
        """幂运算右结合: 2 ^ 3 ^ 4 应为 2 ^ (3 ^ 4)。"""
        ast = parse_expr("2 ^ 3 ^ 4")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "^"
        assert isinstance(ast.right, BinaryOpNode)
        assert ast.right.op == "^"

    def test_comparison_before_logical(self, parse_expr):
        """比较在逻辑之前: $close > 100 && $open < 200。"""
        ast = parse_expr("$close > 100 && $open < 200")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "&&"
        assert isinstance(ast.left, BinaryOpNode)
        assert ast.left.op == ">"

    def test_equality_before_comparison(self, parse_expr):
        """相等在比较之前: $close == $open > $high。"""
        ast = parse_expr("$close == $open > $high")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "=="

    def test_multiple_addition_assoc(self, parse_expr):
        """同级运算左结合: $a + $b + $c。"""
        ast = parse_expr("$close + $open + $high")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "+"
        assert isinstance(ast.left, BinaryOpNode)
        assert ast.left.op == "+"


class TestUnaryOperations:
    """一元运算测试。"""

    def test_unary_negation(self, parse_expr):
        """一元负号。"""
        ast = parse_expr("-$close")
        assert isinstance(ast, UnaryOpNode)
        assert ast.op == "-"
        assert isinstance(ast.operand, VarNode)

    def test_unary_logical_not(self, parse_expr):
        """逻辑非。"""
        ast = parse_expr("!$close")
        assert isinstance(ast, UnaryOpNode)
        assert ast.op == "!"
        assert isinstance(ast.operand, VarNode)

    def test_double_negation(self, parse_expr):
        """双重否定 (应抵消)。"""
        ast = parse_expr("--$close")
        assert isinstance(ast, UnaryOpNode)
        assert ast.op == "-"
        assert isinstance(ast.operand, UnaryOpNode)


class TestFunctionCalls:
    """函数调用测试。"""

    def test_simple_function(self, parse_expr):
        """单参数函数调用。"""
        ast = parse_expr("RANK($close)")
        assert isinstance(ast, FuncCallNode)
        assert ast.name == "RANK"
        assert len(ast.args) == 1
        assert isinstance(ast.args[0], VarNode)

    def test_two_arg_function(self, parse_expr):
        """双参数函数调用。"""
        ast = parse_expr("MEAN($close, 20)")
        assert isinstance(ast, FuncCallNode)
        assert ast.name == "MEAN"
        assert len(ast.args) == 2

    def test_three_arg_function(self, parse_expr):
        """三参数函数调用。"""
        ast = parse_expr("IF($close > 100, $close, 0)")
        assert isinstance(ast, ConditionalNode)

    def test_nested_functions(self, parse_expr):
        """嵌套函数调用。"""
        ast = parse_expr("RANK(DELTA($close, 5))")
        assert isinstance(ast, FuncCallNode)
        assert ast.name == "RANK"
        assert len(ast.args) == 1
        inner = ast.args[0]
        assert isinstance(inner, FuncCallNode)
        assert inner.name == "DELTA"

    def test_deeply_nested_functions(self, parse_expr):
        """深层嵌套函数。"""
        ast = parse_expr("RANK(MEAN(DELTA($close, 5), 10))")
        assert isinstance(ast, FuncCallNode)
        inner = ast.args[0]
        assert isinstance(inner, FuncCallNode)
        assert inner.name == "MEAN"

    def test_function_expression_args(self, parse_expr):
        """函数参数为表达式。"""
        ast = parse_expr("MEAN($close + $open, 20)")
        assert isinstance(ast, FuncCallNode)
        assert isinstance(ast.args[0], BinaryOpNode)

    def test_zero_arg_function(self, parse_expr):
        """零参数函数。"""
        ast = parse_expr("INIT()")
        assert isinstance(ast, FuncCallNode)
        assert len(ast.args) == 0


class TestParentheses:
    """括号表达式测试。"""

    def test_simple_parentheses(self, parse_expr):
        """简单括号。"""
        ast = parse_expr("($close + $open)")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "+"

    def test_nested_parentheses(self, parse_expr):
        """嵌套括号。"""
        ast = parse_expr("(($close + $open) * $high)")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "*"
        assert isinstance(ast.left, BinaryOpNode)

    def test_parentheses_override_precedence(self, parse_expr):
        """括号改变优先级: ($close + $open) * $high。"""
        ast = parse_expr("($close + $open) * $high")
        assert isinstance(ast, BinaryOpNode)
        assert ast.op == "*"
        assert isinstance(ast.left, BinaryOpNode)
        assert ast.left.op == "+"

    def test_deeply_nested_parentheses(self, parse_expr):
        """深层嵌套括号。"""
        ast = parse_expr("((($close + 1)))")
        assert isinstance(ast, BinaryOpNode)


class TestConditionalExpressions:
    """条件表达式测试。"""

    def test_ternary_expression(self, parse_expr):
        """三元表达式 (? :)。"""
        ast = parse_expr("$close > $open ? $close : $open")
        assert isinstance(ast, ConditionalNode)
        assert isinstance(ast.condition, BinaryOpNode)
        assert isinstance(ast.then_branch, VarNode)
        assert isinstance(ast.else_branch, VarNode)

    def test_if_function(self, parse_expr):
        """IF 函数形式。"""
        ast = parse_expr("IF($close > 100, $close, 0)")
        assert isinstance(ast, ConditionalNode)
        assert isinstance(ast.condition, BinaryOpNode)

    def test_if_function_no_else(self, parse_expr):
        """IF 函数无 else 分支。"""
        ast = parse_expr("IF($close > 100, $close)")
        assert isinstance(ast, ConditionalNode)
        assert ast.else_branch is None

    def test_nested_conditional(self, parse_expr):
        """嵌套条件表达式。"""
        ast = parse_expr("$close > 100 ? ($close > 150 ? $high : $close) : $low")
        assert isinstance(ast, ConditionalNode)
        assert isinstance(ast.then_branch, ConditionalNode)

    def test_conditional_with_functions(self, parse_expr):
        """条件表达式与函数组合。"""
        ast = parse_expr("RANK($close) > 0.5 ? $high : $low")
        assert isinstance(ast, ConditionalNode)
        assert isinstance(ast.condition, BinaryOpNode)


class TestErrorHandling:
    """错误处理测试。"""

    def test_unexpected_token(self, lexer, parser):
        """意外 token 应抛出 ParseError。"""
        tokens = lexer.tokenize("$close $open")
        with pytest.raises(ParseError):
            parser.parse(tokens)

    def test_missing_right_paren(self, lexer, parser):
        """缺少右括号应抛出 ParseError。"""
        tokens = lexer.tokenize("($close + $open")
        with pytest.raises(ParseError):
            parser.parse(tokens)

    def test_missing_left_paren(self, lexer, parser):
        """缺少左括号。"""
        tokens = lexer.tokenize("$close + $open)")
        with pytest.raises(ParseError):
            parser.parse(tokens)

    def test_missing_colon_in_ternary(self, lexer, parser):
        """三元表达式缺少冒号。"""
        tokens = lexer.tokenize("$close > $open ? $close $open")
        with pytest.raises(ParseError):
            parser.parse(tokens)

    def test_missing_comma_in_if(self, lexer, parser):
        """IF 函数缺少逗号。"""
        tokens = lexer.tokenize("IF($close > 100 $close 0)")
        with pytest.raises(ParseError):
            parser.parse(tokens)

    def test_unexpected_eof(self, lexer, parser):
        """表达式中途结束。"""
        tokens = lexer.tokenize("$close +")
        with pytest.raises(ParseError):
            parser.parse(tokens)

    def test_illegal_token(self, lexer, parser):
        """非法 token 导致解析错误。"""
        tokens = lexer.tokenize("$close @ $open")
        with pytest.raises(ParseError):
            parser.parse(tokens)

    def test_empty_expression(self, lexer, parser):
        """空表达式 (仅 EOF) 应抛出错误。"""
        from smartalpha.core.lexer import Token, TokenType
        tokens = [Token(TokenType.EOF, "", 0)]
        with pytest.raises(ParseError):
            parser.parse(tokens)

    def test_parse_error_has_token_info(self, lexer, parser):
        """ParseError 应包含 token 位置信息。"""
        tokens = lexer.tokenize("$close +")
        try:
            parser.parse(tokens)
            assert False, "应抛出异常"
        except ParseError as e:
            assert "position" in str(e) or "位置" in str(e) or str(e)

    def test_multiple_expressions(self, lexer, parser):
        """多个表达式 (不以运算符连接) 应抛错。"""
        tokens = lexer.tokenize("$close $open")
        with pytest.raises(ParseError):
            parser.parse(tokens)