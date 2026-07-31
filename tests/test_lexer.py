"""词法分析器 ExpressionLexer 测试套件。

覆盖:
- 变量识别 ($close, $open, $VAR_NAME)
- 数字识别 (整数、浮点、科学计数法)
- 运算符识别 (+, -, *, /, %, ^)
- 比较运算符 (>, <, >=, <=, ==, !=)
- 逻辑运算符 (&&, ||, !)
- 条件表达式 (?, :)
- 函数名识别
- 字符串字面量
- 错误处理 (非法字符、空输入等)
"""

import pytest

from smartalpha.core.lexer import ExpressionLexer, Token, TokenType


class TestExpressionLexerBasic:
    """基础功能测试。"""

    def test_empty_input(self, lexer):
        """空字符串应产生 EOF token。"""
        tokens = lexer.tokenize("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_none_default_text(self, lexer):
        """构造函数无参数时应正常工作。"""
        l = ExpressionLexer()
        tokens = l.tokenize()
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_whitespace_only(self, lexer):
        """纯空白字符应仅产生 EOF。"""
        tokens = lexer.tokenize("   \t\n  ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_tokenize_static_method(self, lexer):
        """验证 tokenize 可以覆盖构造函数传入的文本。"""
        l = ExpressionLexer("$close")
        tokens = l.tokenize("42")
        assert tokens[0].type == TokenType.NUMBER


class TestVariableRecognition:
    """变量识别测试。"""

    def test_simple_variable(self, lexer):
        """简单变量 $close。"""
        tokens = lexer.tokenize("$close")
        assert tokens[0].type == TokenType.VAR
        assert tokens[0].value == "$close"
        assert tokens[0].position == 0

    def test_variable_with_underscore(self, lexer):
        """含下划线的变量名。"""
        tokens = lexer.tokenize("$close_price")
        assert tokens[0].type == TokenType.VAR
        assert tokens[0].value == "$close_price"

    def test_variable_with_digits(self, lexer):
        """含数字的变量名。"""
        tokens = lexer.tokenize("$volume_2024")
        assert tokens[0].type == TokenType.VAR
        assert tokens[0].value == "$volume_2024"

    def test_multiple_variables(self, lexer):
        """多个变量以空格分隔。"""
        tokens = lexer.tokenize("$close $open $high")
        assert tokens[0].type == TokenType.VAR
        assert tokens[0].value == "$close"
        assert tokens[1].type == TokenType.VAR
        assert tokens[1].value == "$open"
        assert tokens[2].type == TokenType.VAR
        assert tokens[2].value == "$high"

    def test_single_dollar_sign(self, lexer):
        """单独的 $ 应产生 ILLEGAL。"""
        tokens = lexer.tokenize("$")
        assert tokens[0].type == TokenType.ILLEGAL
        assert tokens[0].value == "$"

    def test_dollar_then_digit(self, lexer):
        """$ 后跟数字应产生 ILLEGAL。"""
        tokens = lexer.tokenize("$123")
        assert tokens[0].type == TokenType.ILLEGAL

    def test_variable_with_operator(self, lexer):
        """变量与运算符组合。"""
        tokens = lexer.tokenize("$close + $open")
        assert tokens[0].type == TokenType.VAR
        assert tokens[1].type == TokenType.PLUS
        assert tokens[2].type == TokenType.VAR


class TestNumberRecognition:
    """数字识别测试。"""

    def test_integer(self, lexer):
        """整数识别。"""
        tokens = lexer.tokenize("42")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "42"

    def test_float(self, lexer):
        """浮点数识别。"""
        tokens = lexer.tokenize("3.14")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "3.14"

    def test_scientific_notation_e(self, lexer):
        """科学计数法 (e)。"""
        tokens = lexer.tokenize("1e5")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "1e5"

    def test_scientific_notation_E(self, lexer):
        """科学计数法 (E)。"""
        tokens = lexer.tokenize("1E5")
        assert tokens[0].type == TokenType.NUMBER

    def test_scientific_notation_negative_exponent(self, lexer):
        """科学计数法 (负指数)。"""
        tokens = lexer.tokenize("1.5e-10")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "1.5e-10"

    def test_scientific_notation_positive_exponent(self, lexer):
        """科学计数法 (正指数显式)。"""
        tokens = lexer.tokenize("2.0e+3")
        assert tokens[0].type == TokenType.NUMBER

    def test_multiple_dots_number(self, lexer):
        """多个小数点: 1.2 被识别，.3 作为另一个 NUMBER。"""
        tokens = lexer.tokenize("1.2.3")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "1.2"
        assert tokens[1].type == TokenType.NUMBER
        assert tokens[1].value == ".3"

    def test_zero(self, lexer):
        """零值。"""
        tokens = lexer.tokenize("0")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "0"

    def test_large_number(self, lexer):
        """大数识别。"""
        tokens = lexer.tokenize("1000000000")
        assert tokens[0].type == TokenType.NUMBER

    def test_small_float(self, lexer):
        """小型浮点数。"""
        tokens = lexer.tokenize("0.001")
        assert tokens[0].type == TokenType.NUMBER

    def test_number_followed_by_operator(self, lexer):
        """数字后跟运算符。"""
        tokens = lexer.tokenize("42+1")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "42"
        assert tokens[1].type == TokenType.PLUS


class TestArithmeticOperators:
    """算术运算符测试。"""

    @pytest.mark.parametrize("op_char,expected_type", [
        ("+", TokenType.PLUS),
        ("-", TokenType.MINUS),
        ("*", TokenType.STAR),
        ("/", TokenType.SLASH),
        ("%", TokenType.PERCENT),
        ("^", TokenType.CARET),
    ])
    def test_single_arithmetic_operator(self, lexer, op_char, expected_type):
        """单个算术运算符。"""
        tokens = lexer.tokenize(op_char)
        assert tokens[0].type == expected_type
        assert tokens[0].value == op_char

    def test_addition(self, lexer):
        """加法。"""
        tokens = lexer.tokenize("$close + 1")
        assert tokens[1].type == TokenType.PLUS

    def test_subtraction(self, lexer):
        """减法。"""
        tokens = lexer.tokenize("$close - $open")
        assert tokens[1].type == TokenType.MINUS

    def test_multiplication(self, lexer):
        """乘法。"""
        tokens = lexer.tokenize("2 * $close")
        assert tokens[1].type == TokenType.STAR

    def test_division(self, lexer):
        """除法。"""
        tokens = lexer.tokenize("$close / $open")
        assert tokens[1].type == TokenType.SLASH

    def test_modulo(self, lexer):
        """取模。"""
        tokens = lexer.tokenize("$close % 3")
        assert tokens[1].type == TokenType.PERCENT

    def test_power(self, lexer):
        """幂运算。"""
        tokens = lexer.tokenize("$close ^ 2")
        assert tokens[1].type == TokenType.CARET


class TestComparisonOperators:
    """比较运算符测试。"""

    def test_greater_than(self, lexer):
        """大于。"""
        tokens = lexer.tokenize("$close > $open")
        assert tokens[1].type == TokenType.GT
        assert tokens[1].value == ">"

    def test_greater_equal(self, lexer):
        """大于等于。"""
        tokens = lexer.tokenize("$close >= 100")
        assert tokens[1].type == TokenType.GTE
        assert tokens[1].value == ">="

    def test_less_than(self, lexer):
        """小于。"""
        tokens = lexer.tokenize("$close < $open")
        assert tokens[1].type == TokenType.LT
        assert tokens[1].value == "<"

    def test_less_equal(self, lexer):
        """小于等于。"""
        tokens = lexer.tokenize("$close <= 200")
        assert tokens[1].type == TokenType.LTE
        assert tokens[1].value == "<="

    def test_equal(self, lexer):
        """等于。"""
        tokens = lexer.tokenize("$close == $open")
        assert tokens[1].type == TokenType.EQ
        assert tokens[1].value == "=="

    def test_not_equal(self, lexer):
        """不等于。"""
        tokens = lexer.tokenize("$close != $open")
        assert tokens[1].type == TokenType.NEQ
        assert tokens[1].value == "!="


class TestLogicalOperators:
    """逻辑运算符测试。"""

    def test_and_operator(self, lexer):
        """逻辑与 &&。"""
        tokens = lexer.tokenize("$close > 100 && $volume > 1000")
        assert tokens[3].type == TokenType.AND
        assert tokens[3].value == "&&"

    def test_or_operator(self, lexer):
        """逻辑或 ||。"""
        tokens = lexer.tokenize("$close > 100 || $high < 200")
        assert tokens[3].type == TokenType.OR
        assert tokens[3].value == "||"

    def test_not_operator(self, lexer):
        """逻辑非 !。"""
        tokens = lexer.tokenize("!$close")
        assert tokens[0].type == TokenType.NOT
        assert tokens[0].value == "!"

    def test_complex_logical(self, lexer):
        """复杂逻辑组合。"""
        tokens = lexer.tokenize("($close > 100) && ($open < 200) || $volume > 1000")
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        assert TokenType.AND in types
        assert TokenType.OR in types


class TestConditionalExpression:
    """条件表达式测试。"""

    def test_question_mark(self, lexer):
        """问号识别。"""
        tokens = lexer.tokenize("$close > $open ? $close : $open")
        assert tokens[3].type == TokenType.QUESTION
        assert tokens[3].value == "?"

    def test_colon(self, lexer):
        """冒号识别。"""
        tokens = lexer.tokenize("$close > $open ? $close : $open")
        assert tokens[5].type == TokenType.COLON
        assert tokens[5].value == ":"

    def test_ternary_structure(self, lexer):
        """三元表达式结构验证。"""
        tokens = lexer.tokenize("$close > 100 ? $high : $low")
        types = [t.type for t in tokens[:-1]]
        assert types == [
            TokenType.VAR, TokenType.GT, TokenType.NUMBER,
            TokenType.QUESTION, TokenType.VAR, TokenType.COLON, TokenType.VAR,
        ]


class TestFunctionRecognition:
    """函数名识别测试。"""

    def test_simple_function_call(self, lexer):
        """简单函数调用。"""
        tokens = lexer.tokenize("RANK($close)")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "RANK"
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.VAR
        assert tokens[3].type == TokenType.RPAREN

    def test_function_with_multiple_args(self, lexer):
        """多参数函数调用。"""
        tokens = lexer.tokenize("MEAN($close, 20)")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "MEAN"
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.VAR
        assert tokens[3].type == TokenType.COMMA
        assert tokens[4].type == TokenType.NUMBER

    def test_nested_functions(self, lexer):
        """嵌套函数调用。"""
        tokens = lexer.tokenize("RANK(DELTA($close, 5))")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "RANK"
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "DELTA"

    def test_underscore_function_name(self, lexer):
        """带下划线的函数名。"""
        tokens = lexer.tokenize("MY_FUNC($close)")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "MY_FUNC"

    def test_zero_arg_function(self, lexer):
        """无参数函数调用。"""
        tokens = lexer.tokenize("INIT()")
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.RPAREN


class TestKeywords:
    """关键字识别测试。"""

    def test_if_keyword(self, lexer):
        """IF 关键字。"""
        tokens = lexer.tokenize("IF($close > 100, $close, 0)")
        assert tokens[0].type == TokenType.IF
        assert tokens[0].value == "IF"

    def test_else_keyword(self, lexer):
        """ELSE 关键字在表达式中作为标识符。"""
        tokens = lexer.tokenize("else")
        assert tokens[0].type == TokenType.ELSE

    def test_if_lowercase(self, lexer):
        """小写 if 应识别为 IF。"""
        tokens = lexer.tokenize("if($close > 100, $close, 0)")
        assert tokens[0].type == TokenType.IF

    def test_true_keyword(self, lexer):
        """TRUE 关键字应转为 NUMBER(1)。"""
        tokens = lexer.tokenize("TRUE")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "1"

    def test_false_keyword(self, lexer):
        """FALSE 关键字应转为 NUMBER(0)。"""
        tokens = lexer.tokenize("FALSE")
        assert tokens[0].type == TokenType.NUMBER
        assert tokens[0].value == "0"


class TestStringLiterals:
    """字符串字面量测试。"""

    def test_double_quoted_string(self, lexer):
        """双引号字符串。"""
        tokens = lexer.tokenize('"hello"')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == '"hello"'

    def test_single_quoted_string(self, lexer):
        """单引号字符串。"""
        tokens = lexer.tokenize("'world'")
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == "'world'"

    def test_string_with_escape(self, lexer):
        """带转义字符的字符串。"""
        tokens = lexer.tokenize('"hello\\nworld"')
        assert tokens[0].type == TokenType.STRING

    def test_empty_string(self, lexer):
        """空字符串。"""
        tokens = lexer.tokenize('""')
        assert tokens[0].type == TokenType.STRING
        assert tokens[0].value == '""'


class TestDelimiters:
    """分隔符测试。"""

    def test_left_paren(self, lexer):
        """左括号。"""
        tokens = lexer.tokenize("(")
        assert tokens[0].type == TokenType.LPAREN

    def test_right_paren(self, lexer):
        """右括号。"""
        tokens = lexer.tokenize(")")
        assert tokens[0].type == TokenType.RPAREN

    def test_comma(self, lexer):
        """逗号。"""
        tokens = lexer.tokenize(",")
        assert tokens[0].type == TokenType.COMMA


class TestComplexExpressions:
    """复杂表达式测试。"""

    def test_arithmetic_precedence_tokens(self, lexer):
        """算术运算组合。"""
        tokens = lexer.tokenize("$close + $open * $high")
        assert len([t for t in tokens if t.type != TokenType.EOF]) == 5

    def test_nested_parentheses(self, lexer):
        """嵌套括号。"""
        tokens = lexer.tokenize("(($close + $open) * $high)")
        types = [t.type for t in tokens[:-1]]
        assert types.count(TokenType.LPAREN) == 2
        assert types.count(TokenType.RPAREN) == 2

    def test_full_alpha_expression(self, lexer):
        """完整 Alpha158 风格表达式。"""
        expr = "RANK(DELTA($close, 5)) / STD($volume, 20)"
        tokens = lexer.tokenize(expr)
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "RANK"
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "DELTA"

    def test_conditional_with_functions(self, lexer):
        """条件表达式与函数组合。"""
        expr = "IF($close > MA($close, 20), $high, $low)"
        tokens = lexer.tokenize(expr)
        assert tokens[0].type == TokenType.IF

    def test_multiple_operators(self, lexer):
        """连续运算符组合。"""
        tokens = lexer.tokenize("$close + $open - $high * $low / $volume")
        types = [t.type for t in tokens[:-1]]
        assert types.count(TokenType.PLUS) == 1
        assert types.count(TokenType.MINUS) == 1
        assert types.count(TokenType.STAR) == 1
        assert types.count(TokenType.SLASH) == 1


class TestErrorHandling:
    """错误处理测试。"""

    def test_invalid_character(self, lexer):
        """非法字符产生 ILLEGAL token。"""
        tokens = lexer.tokenize("$close @ $open")
        illegal_tokens = [t for t in tokens if t.type == TokenType.ILLEGAL]
        assert len(illegal_tokens) >= 1
        assert illegal_tokens[0].value == "@"

    def test_hash_character(self, lexer):
        """# 字符应为 ILLEGAL。"""
        tokens = lexer.tokenize("#test")
        assert tokens[0].type == TokenType.ILLEGAL

    def test_at_sign(self, lexer):
        """@ 符号应为 ILLEGAL。"""
        tokens = lexer.tokenize("@")
        assert tokens[0].type == TokenType.ILLEGAL

    def test_special_chars(self, lexer):
        """多个特殊字符。"""
        tokens = lexer.tokenize("$close & $open | $high")
        illegal_count = len([t for t in tokens if t.type == TokenType.ILLEGAL])
        assert illegal_count >= 2

    def test_position_tracking(self, lexer):
        """验证 token 位置信息。"""
        tokens = lexer.tokenize("  $close  +  42")
        assert tokens[0].position == 2
        assert tokens[1].position == 10
        assert tokens[2].position == 13

    def test_token_repr(self, lexer):
        """验证 Token 的 repr。"""
        token = Token(TokenType.VAR, "$close", 0)
        r = repr(token)
        assert "VAR" in r
        assert "$close" in r

    def test_case_insensitive_keywords(self, lexer):
        """大小写不敏感关键字。"""
        tokens_upper = lexer.tokenize("IF")
        tokens_lower = lexer.tokenize("if")
        assert tokens_upper[0].type == TokenType.IF
        assert tokens_lower[0].type == TokenType.IF

    def test_duplicate_lexer_state(self, lexer):
        """验证多次调用 tokenize 状态正确重置。"""
        tokens1 = lexer.tokenize("$close")
        tokens2 = lexer.tokenize("$open")
        assert tokens1[0].value == "$close"
        assert tokens2[0].value == "$open"

    def test_long_identifier(self, lexer):
        """长标识符。"""
        long_name = "A" * 200
        tokens = lexer.tokenize(long_name)
        assert tokens[0].type == TokenType.IDENTIFIER
        assert len(tokens[0].value) == 200