"""AST 执行器 ASTExecutor 测试套件。

覆盖:
- 变量求值
- 算术运算 (加、减、乘、除、取模、幂)
- 比较运算
- 逻辑运算
- 一元运算
- 函数调用执行
- 条件表达式执行
- 复杂嵌套表达式
- 错误处理
"""

import numpy as np
import pytest

from smartalpha.core.lexer import ExpressionLexer
from smartalpha.core.parser import ExpressionParser
from smartalpha.core.executor import ASTExecutor, ExecutionError
from smartalpha.core.functions import FinancialFunctionLibrary


@pytest.fixture
def execute_expr(lexer, parser):
    """便捷方法: 解析并执行表达式。自动为变量键添加 $ 前缀。"""
    def _execute(expr: str, variables: dict | None = None):
        if variables:
            processed = {}
            for k, v in variables.items():
                if not k.startswith("$"):
                    processed[f"${k}"] = v
                else:
                    processed[k] = v
        else:
            processed = {}
        tokens = lexer.tokenize(expr)
        ast = parser.parse(tokens)
        executor = ASTExecutor(variables=processed)
        return executor.execute(ast)
    return _execute


class TestVariableEvaluation:
    """变量求值测试。"""

    def test_simple_variable(self, execute_expr, small_data):
        """简单变量求值。"""
        result = execute_expr("$close", small_data)
        np.testing.assert_array_almost_equal(result, small_data["close"])

    def test_multiple_variables(self, execute_expr, small_data):
        """多变量在同一表达式中。"""
        result = execute_expr("$close + $open", small_data)
        np.testing.assert_array_almost_equal(
            result, small_data["close"] + small_data["open"]
        )

    def test_undefined_variable(self, execute_expr):
        """未定义变量应抛出 ExecutionError。"""
        with pytest.raises(ExecutionError):
            execute_expr("$undefined_var", {})

    def test_variable_scalar(self, execute_expr):
        """标量变量。"""
        result = execute_expr("$x", {"$x": 42.0})
        assert float(result) == 42.0


class TestArithmeticOperations:
    """算术运算测试。"""

    def test_addition(self, execute_expr, small_data):
        """加法。"""
        result = execute_expr("$close + $open", small_data)
        expected = small_data["close"] + small_data["open"]
        np.testing.assert_array_almost_equal(result, expected)

    def test_subtraction(self, execute_expr, small_data):
        """减法。"""
        result = execute_expr("$close - $open", small_data)
        expected = small_data["close"] - small_data["open"]
        np.testing.assert_array_almost_equal(result, expected)

    def test_multiplication(self, execute_expr, small_data):
        """乘法。"""
        result = execute_expr("$close * $open", small_data)
        expected = small_data["close"] * small_data["open"]
        np.testing.assert_array_almost_equal(result, expected)

    def test_division(self, execute_expr, small_data):
        """除法。"""
        result = execute_expr("$close / $open", small_data)
        expected = small_data["close"] / small_data["open"]
        np.testing.assert_array_almost_equal(result, expected)

    def test_division_by_zero(self, execute_expr):
        """除零应返回 0 (已在 executor 中处理)。"""
        result = execute_expr("$x / $y", {"$x": np.array([10.0, 20.0]), "$y": np.array([0.0, 0.0])})
        np.testing.assert_array_almost_equal(result, np.array([0.0, 0.0]))

    def test_modulo(self, execute_expr):
        """取模。"""
        result = execute_expr("$x % $y", {"$x": np.array([10.0, 11.0]), "$y": np.array([3.0, 3.0])})
        expected = np.array([1.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_power(self, execute_expr):
        """幂运算。"""
        result = execute_expr("$x ^ 2", {"$x": np.array([2.0, 3.0, 4.0])})
        expected = np.array([4.0, 9.0, 16.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_scalar_operations(self, execute_expr):
        """标量与数组运算。"""
        result = execute_expr("$x * 2", {"$x": np.array([1.0, 2.0, 3.0])})
        np.testing.assert_array_almost_equal(result, np.array([2.0, 4.0, 6.0]))

    def test_chain_operations(self, execute_expr, sample_1d):
        """链式运算。"""
        result = execute_expr("$close + $open - $high", sample_1d)
        expected = sample_1d["close"] + sample_1d["open"] - sample_1d["high"]
        np.testing.assert_array_almost_equal(result, expected)


class TestComparisonOperations:
    """比较运算测试。"""

    def test_greater_than(self, execute_expr, small_data):
        """大于比较。"""
        result = execute_expr("$close > $open", small_data)
        expected = (small_data["close"] > small_data["open"]).astype(np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_less_than(self, execute_expr, sample_1d):
        """小于比较。"""
        result = execute_expr("$close < $high", sample_1d)
        expected = (sample_1d["close"] < sample_1d["high"]).astype(np.float64)
        np.testing.assert_array_almost_equal(result, expected)

    def test_equal_comparison(self, execute_expr):
        """等于比较。"""
        result = execute_expr("$x == $y", {"$x": np.array([1.0, 2.0, 3.0]), "$y": np.array([1.0, 0.0, 3.0])})
        expected = np.array([1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_not_equal(self, execute_expr):
        """不等于比较。"""
        result = execute_expr("$x != $y", {"$x": np.array([1.0, 2.0]), "$y": np.array([1.0, 0.0])})
        expected = np.array([0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestLogicalOperations:
    """逻辑运算测试。"""

    def test_logical_and(self, execute_expr):
        """逻辑与。"""
        result = execute_expr("$x > 0 && $y > 0", {
            "$x": np.array([1.0, -1.0, 2.0]),
            "$y": np.array([1.0, 1.0, -1.0]),
        })
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_logical_or(self, execute_expr):
        """逻辑或。"""
        result = execute_expr("$x > 0 || $y > 0", {
            "$x": np.array([1.0, -1.0, -1.0]),
            "$y": np.array([-1.0, -1.0, 2.0]),
        })
        expected = np.array([1.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestUnaryOperations:
    """一元运算测试。"""

    def test_negation(self, execute_expr):
        """一元负号。"""
        result = execute_expr("-$x", {"$x": np.array([1.0, -2.0, 3.0])})
        expected = np.array([-1.0, 2.0, -3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_logical_not(self, execute_expr):
        """逻辑非。"""
        result = execute_expr("!$x", {"$x": np.array([1.0, 0.0, 1.0])})
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestFunctionExecution:
    """函数调用执行测试。"""

    def test_rank_function(self, execute_expr, small_data):
        """RANK 函数。"""
        result = execute_expr("RANK($close)", small_data)
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        assert len(result) == len(small_data["close"])

    def test_mean_function(self, execute_expr, small_data):
        """MEAN 函数 (全局)。"""
        result = execute_expr("MEAN($close)", small_data)
        expected = np.full_like(small_data["close"], np.nanmean(small_data["close"]))
        np.testing.assert_array_almost_equal(result, expected)

    def test_mean_window(self, execute_expr, small_data):
        """MEAN 函数 (滚动窗口)。"""
        result = execute_expr("MEAN($close, 3)", small_data)
        assert len(result) == len(small_data["close"])
        assert not np.isnan(result[-1])

    def test_delta_function(self, execute_expr, small_data):
        """DELTA 函数。"""
        result = execute_expr("DELTA($close, 1)", small_data)
        assert len(result) == len(small_data["close"])

    def test_unknown_function(self, execute_expr, small_data):
        """未知函数应抛出 ExecutionError。"""
        with pytest.raises(ExecutionError):
            execute_expr("UNKNOWN_FUNC($close)", small_data)


class TestConditionalExecution:
    """条件表达式执行测试。"""

    def test_ternary_conditional(self, execute_expr, small_data):
        """三元条件表达式。"""
        result = execute_expr("$close > $open ? $close : $open", small_data)
        assert len(result) == len(small_data["close"])
        for i in range(len(result)):
            if small_data["close"][i] > small_data["open"][i]:
                assert result[i] == small_data["close"][i]
            else:
                assert result[i] == small_data["open"][i]

    def test_if_function(self, execute_expr, sample_1d):
        """IF 函数。"""
        result = execute_expr("IF($close > $open, $high, $low)", sample_1d)
        assert len(result) == len(sample_1d["close"])

    def test_if_without_else(self, execute_expr, sample_1d):
        """IF 函数无 else 分支。"""
        result = execute_expr("IF($close > $open, $high)", sample_1d)
        assert len(result) == len(sample_1d["close"])

    def test_nested_conditional(self, execute_expr, sample_1d):
        """嵌套条件。"""
        result = execute_expr(
            "$close > 12 ? ($close > 14 ? $high : $close) : $low",
            sample_1d,
        )
        assert len(result) == len(sample_1d["close"])


class TestComplexExpressions:
    """复杂嵌套表达式测试。"""

    def test_nested_functions(self, execute_expr, small_data):
        """嵌套函数调用。"""
        result = execute_expr("RANK(DELTA($close, 1))", small_data)
        assert len(result) == len(small_data["close"])

    def test_function_with_conditional(self, execute_expr, small_data):
        """函数与条件组合。"""
        result = execute_expr(
            "MEAN(IF($close > $open, $close, $open), 5)",
            small_data,
        )
        assert len(result) == len(small_data["close"])

    def test_arithmetic_with_functions(self, execute_expr, small_data):
        """算术与函数组合。"""
        result = execute_expr("RANK($close) + MEAN($volume, 5)", small_data)
        assert len(result) == len(small_data["close"])

    def test_complex_alpha_expression(self, execute_expr, sample_1d):
        """Alpha158 风格复杂表达式。"""
        result = execute_expr(
            "RANK(DELTA($close, 5)) / STD($volume, 20)",
            sample_1d,
        )
        assert len(result) == len(sample_1d["close"])
        valid = ~np.isnan(result)
        assert np.sum(valid) > 0

    def test_expression_with_constants(self, execute_expr, small_data):
        """含常量的表达式。"""
        result = execute_expr("$close * 1.05", small_data)
        expected = small_data["close"] * 1.05
        np.testing.assert_array_almost_equal(result, expected)

    def test_precedence_in_execution(self, execute_expr, sample_1d):
        """执行时运算符优先级。"""
        result = execute_expr("$close + $open * $high", sample_1d)
        expected = sample_1d["close"] + sample_1d["open"] * sample_1d["high"]
        np.testing.assert_array_almost_equal(result, expected)

    def test_conditional_involving_functions(self, execute_expr, sample_1d):
        """条件表达式中嵌入函数。"""
        result = execute_expr(
            "RANK($close) > 0.5 ? $high : $low",
            sample_1d,
        )
        assert len(result) == len(sample_1d["close"])


class TestExecutorEdgeCases:
    """执行器边界条件。"""

    def test_scalar_result(self, execute_expr):
        """纯标量表达式。"""
        result = execute_expr("42", {})
        assert float(result) == 42.0

    def test_negative_number(self, execute_expr):
        """负数。"""
        result = execute_expr("-42", {})
        assert float(result) == -42.0

    def test_executor_constructor_variables(self, lexer, parser):
        """通过构造函数传入变量。"""
        tokens = lexer.tokenize("$x + $y")
        ast = parser.parse(tokens)
        executor = ASTExecutor(variables={"$x": 10.0, "$y": 20.0})
        result = executor.execute(ast)
        assert float(result) == 30.0

    def test_execute_with_runtime_variables(self, lexer, parser):
        """通过 execute 方法覆盖变量。"""
        tokens = lexer.tokenize("$x + $y")
        ast = parser.parse(tokens)
        executor = ASTExecutor()
        result = executor.execute(ast, {"$x": 5.0, "$y": 15.0})
        assert float(result) == 20.0

    def test_executor_custom_functions(self, lexer, parser):
        """自定义函数执行。"""
        lib = FinancialFunctionLibrary()
        lib.register("DOUBLE", lambda x: x * 2)
        tokens = lexer.tokenize("DOUBLE($x)")
        ast = parser.parse(tokens)
        executor = ASTExecutor(variables={"$x": np.array([1.0, 2.0, 3.0])}, functions=lib)
        result = executor.execute(ast)
        np.testing.assert_array_almost_equal(result, np.array([2.0, 4.0, 6.0]))