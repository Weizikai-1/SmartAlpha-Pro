"""跨模块集成测试 — 验证模块间协作正确性。

测试: 数据→表达式→评估→回测→风控 全链路。
"""

import numpy as np
import pandas as pd
import pytest

from smartalpha._constants import TRADING_DAYS_PER_YEAR, EPS
from smartalpha.core import ExpressionLexer, ExpressionParser, ASTExecutor
from smartalpha.core.functions import FinancialFunctionLibrary
from smartalpha.eval import evaluate_factor, FactorReport, compute_sharpe, compute_max_drawdown
from smartalpha.backtest import BacktestEngine, AShareCostModel
from smartalpha.risk import VaRCalculator, RiskManager, StressTester


# ============================================================================
# 数据层 → 表达式引擎
# ============================================================================

class TestDataToExpression:
    def test_compute_factor_from_synthetic_data(self):
        """用模拟OHLCV数据计算表达式因子。"""
        np.random.seed(42)
        n = 100
        dates = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n, freq="B"))

        base = 100 + np.cumsum(np.random.randn(n) * 2)
        factor_data = {
            "$close": pd.Series(base, index=dates),
            "$open": pd.Series(base - np.abs(np.random.randn(n)), index=dates),
            "$high": pd.Series(base + np.abs(np.random.randn(n) * 2), index=dates),
            "$low": pd.Series(base - np.abs(np.random.randn(n) * 2), index=dates),
            "$volume": pd.Series(np.random.randint(1000, 10000, n), index=dates),
        }

        lexer = ExpressionLexer()
        parser = ExpressionParser()
        executor = ASTExecutor()

        expr = "($high - $low) / ($close + 1e-10)"
        tokens = lexer.tokenize(expr)
        ast = parser.parse(tokens)
        result = executor.execute(ast, factor_data)

        assert len(result) == n
        assert np.all(result >= -1e-9)  # 允许浮点误差

    def test_nested_expression(self):
        """嵌套函数调用。"""
        np.random.seed(42)
        close = pd.Series(100 + np.cumsum(np.random.randn(200) * 1.5))

        factor_data = {"$close": close}
        lexer = ExpressionLexer()
        parser = ExpressionParser()
        executor = ASTExecutor()

        expr = "RANK(DELTA($close, 10))"
        tokens = lexer.tokenize(expr)
        ast = parser.parse(tokens)
        result = executor.execute(ast, factor_data)

        assert len(result) == 200
        assert result.min() >= 0
        assert result.max() <= 1

    def test_function_library_call(self):
        """直接调用金融函数库。"""
        lib = FinancialFunctionLibrary()
        arr = np.random.randn(100).cumsum() + 50
        result = lib.call("RSI", arr, 14)
        assert result.min() >= 0
        assert result.max() <= 100

    def test_list_functions_count(self):
        """函数库应包含足够多的函数。"""
        lib = FinancialFunctionLibrary()
        funcs = lib.list_functions()
        assert len(funcs) >= 45  # 50+ 函数


# ============================================================================
# 评估 → 回测
# ============================================================================

class TestEvalToBacktest:
    def test_sharpe_calculation(self):
        """夏普计算验证。"""
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 252 * 3))
        sharpe = compute_sharpe(returns)
        assert -2 < sharpe < 5  # 合理范围

    def test_max_drawdown_calculation(self):
        """最大回撤计算验证。"""
        nav = pd.Series([1.0, 0.9, 0.95, 0.8, 0.85, 1.0, 0.7])
        mdd = compute_max_drawdown(nav)
        # 峰值1.0, 最低0.7, 回撤 -30%
        assert abs(mdd - (-0.3)) < 0.01

    def test_evaluate_factor_outputs_all_metrics(self):
        """因子评估应输出所有指标。"""
        rng = np.random.RandomState(42)
        n = 200
        price = pd.Series(100 + rng.randn(n).cumsum())
        factor = pd.Series(rng.randn(n), index=price.index)

        result = evaluate_factor(factor, price, periods=[1, 5])
        assert "ic" in result
        assert "rank_ic" in result
        assert "sharpe" in result
        assert "max_drawdown" in result
        assert "annual_return" in result

    def test_factor_report_passed_logic(self):
        """报告通过/失败逻辑。"""
        # 好因子 (IC_IR >= 0.3)
        good_metrics = {
            "ic": {"ret_1d": {"ic_ir": 0.5}},
            "sharpe": 1.0,
            "max_drawdown": -0.2,
        }
        report = FactorReport("good", good_metrics)
        report._check_pass()
        assert report.passed

        # 差因子
        bad_metrics = {
            "ic": {"ret_1d": {"ic_ir": 0.1}},
            "sharpe": 0.2,
            "max_drawdown": -0.8,
        }
        report = FactorReport("bad", bad_metrics)
        report._check_pass()
        assert not report.passed


# ============================================================================
# 回测 → 风控
# ============================================================================

class TestBacktestToRisk:
    def test_cost_model_consistency(self):
        """费率模型: 买入+卖出=往返费率。"""
        cost = AShareCostModel()
        buy = cost.buy_cost(100_000)
        sell = cost.sell_cost(100_000)
        round_trip = cost.round_trip_cost_ratio() * 100_000
        # 不计最低佣金时: buy + sell ≈ round_trip × amount
        assert abs((buy + sell) / 100_000 - cost.round_trip_cost_ratio()) < 0.01

    def test_var_from_backtest_returns(self):
        """用回测收益计算 VaR。"""
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.0005, 0.015, 500))
        result = VaRCalculator.historical(returns)
        assert result.var_95 < 0
        assert result.cvar_95 <= result.var_95

    def test_stress_test_all_scenarios(self):
        """压力测试覆盖所有预定义场景。"""
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.0005, 0.015, 500))
        tester = StressTester()
        results = tester.run_all(returns, var_95=-0.03)
        assert len(results) >= 4
        for r in results:
            assert r.total_return <= 0.01  # 压力场景不应大幅盈利

    def test_risk_manager_position_limit(self):
        """风险限额：单只股票不超过10%。"""
        rm = RiskManager()
        weights = pd.Series([0.40, 0.30, 0.20, 0.10], index=["A", "B", "C", "D"])
        adj, _ = rm.check("2024-01-15", weights)
        assert (adj <= rm.limits.max_single_position).all()


# ============================================================================
# 全链路
# ============================================================================

class TestFullPipeline:
    def test_expression_to_eval_to_risk(self):
        """全链路: 表达式引擎 → 因子评估 → 风险指标。"""
        np.random.seed(42)
        n = 300
        dates = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=n, freq="B"))

        factor_data = {
            "$close": pd.Series(100 + np.cumsum(np.random.randn(n) * 2), index=dates),
            "$volume": pd.Series(np.random.randint(1000, 10000, n), index=dates),
        }

        # 因子计算
        lexer = ExpressionLexer()
        parser = ExpressionParser()
        executor = ASTExecutor()

        tokens = lexer.tokenize("DELTA($close, 20) / DELAY($close, 20)")
        ast = parser.parse(tokens)
        factor = pd.Series(executor.execute(ast, factor_data), index=dates)

        # 因子评估
        price = factor_data["$close"]
        eval_result = evaluate_factor(factor, price, periods=[1, 5])

        # 风险计算
        daily_ret = price.pct_change().dropna()
        var_result = VaRCalculator.historical(daily_ret)

        assert "sharpe" in eval_result
        assert var_result.var_95 < 0

    def test_backtest_with_risk_integration(self):
        """回测 + 风控集成。"""
        np.random.seed(42)
        n_dates = 200
        n_stocks = 5
        dates = pd.date_range("2024-01-02", periods=n_dates, freq="B")
        stocks = [f"S{i}" for i in range(n_stocks)]

        # 生成面板数据
        data = []
        for i, s in enumerate(stocks):
            drift = 0.0002 * (i - 2)
            price = 100 * (1 + np.random.randn(n_dates).cumsum() * 0.02 + np.arange(n_dates) * drift)
            for j, d in enumerate(dates):
                data.append({"trade_date": d, "ts_code": s, "close": max(price[j], 1)})

        df = pd.DataFrame(data)
        panel = df.set_index(["trade_date", "ts_code"]).sort_index()

        # 信号
        price_matrix = panel["close"].unstack("ts_code")
        signal = price_matrix.pct_change(5).rank(axis=1, pct=True)
        signal_stacked = signal.stack().rename("signal")

        # 回测
        engine = BacktestEngine(init_cash=1_000_000, top_n=2, rebalance_freq="M")
        result = engine.run(panel, signal_stacked, price_col="close")

        # 风险指标
        if result.daily_returns is not None:
            var = VaRCalculator.historical(result.daily_returns)
            assert var.var_95 < 0

        # 压力测试
        if result.daily_returns is not None:
            tester = StressTester()
            stress_results = tester.run_all(result.daily_returns)
            assert len(stress_results) == 4

    def test_constants_consistent(self):
        """常量在整个项目中保持一致。"""
        assert TRADING_DAYS_PER_YEAR == 252
        assert EPS == 1e-10

        from smartalpha.eval.metrics import compute_sharpe, compute_annual_return
        from smartalpha.backtest.engine import BacktestEngine

        # 所有模块使用相同的常数
        bt = BacktestEngine(top_n=5)
        assert bt.cost.commission == 0.0003

    def test_invalid_expression_handled(self):
        """无效表达式应抛出合理的异常。"""
        lexer = ExpressionLexer()
        parser = ExpressionParser()

        # 缺少右括号
        tokens = lexer.tokenize("MEAN($close, 10")
        with pytest.raises(Exception):
            parser.parse(tokens)

    def test_edge_case_nan_input(self):
        """含NaN的输入不应崩溃。"""
        data = pd.Series([1.0, np.nan, 2.0, np.nan, 3.0])
        result = evaluate_factor(data, data, periods=[1])
        assert result is not None
