"""回测引擎单元测试。

测试: 费率模型、选股逻辑、绩效指标、净值计算。
"""

import numpy as np
import pandas as pd
import pytest

from smartalpha.backtest.engine import AShareCostModel, BacktestEngine


# ============================================================================
# 费率模型测试
# ============================================================================

class TestCostModel:
    def test_default_rates(self):
        c = AShareCostModel()
        assert c.commission == 0.0003
        assert c.stamp_duty == 0.0005
        assert c.slippage == 0.001

    def test_round_trip(self):
        c = AShareCostModel()
        ratio = c.round_trip_cost_ratio()
        expected = 2 * 0.0003 + 0.0005 + 2 * 0.001
        assert abs(ratio - expected) < 1e-10

    def test_buy_cost(self):
        c = AShareCostModel()
        # ¥100,000 买入成本 = 佣金30 + 滑点100 = 130
        cost = c.buy_cost(100_000)
        assert cost == pytest.approx(130, rel=0.01)

    def test_buy_cost_min(self):
        c = AShareCostModel()
        # 小额交易受最低佣金限制
        cost = c.buy_cost(100)
        assert cost >= 5.0  # 最低5元
        assert cost == pytest.approx(5.1, rel=0.1)

    def test_sell_cost(self):
        c = AShareCostModel()
        # ¥100,000 卖出成本 = 佣金30 + 印花税50 + 滑点100 = 180
        cost = c.sell_cost(100_000)
        assert cost == pytest.approx(180, rel=0.01)

    def test_custom_rates(self):
        c = AShareCostModel(commission=0.0001, stamp_duty=0.001, slippage=0.0005)
        expected = 2 * 0.0001 + 0.001 + 2 * 0.0005
        assert abs(c.round_trip_cost_ratio() - expected) < 1e-10


# ============================================================================
# 选股逻辑测试
# ============================================================================

class TestSelectStocks:
    @pytest.fixture
    def engine(self):
        return BacktestEngine(top_n=3, cost_model=AShareCostModel())

    def test_top_n_selection(self, engine):
        signal = pd.Series(
            [0.1, 0.5, 0.3, 0.9, 0.2],
            index=["A", "B", "C", "D", "E"],
        )
        stocks = pd.Index(["A", "B", "C", "D", "E"])
        weights = engine._select_stocks(signal, stocks)

        # 应选信号最高的3只 (D=0.9, B=0.5, C=0.3)
        assert weights["D"] > 0
        assert weights["B"] > 0
        assert weights["C"] > 0
        assert weights["A"] == 0
        assert weights["E"] == 0
        # 等权
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_fewer_stocks_than_top_n(self, engine):
        signal = pd.Series([0.5, 0.8], index=["X", "Y"])
        stocks = pd.Index(["X", "Y"])
        weights = engine._select_stocks(signal, stocks)

        assert weights["X"] == 0.5
        assert weights["Y"] == 0.5
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_all_nan_signal(self, engine):
        signal = pd.Series([np.nan, np.nan], index=["X", "Y"])
        stocks = pd.Index(["X", "Y"])
        weights = engine._select_stocks(signal, stocks)

        assert weights.sum() == 0  # 全部为0


# ============================================================================
# 调仓日期测试
# ============================================================================

class TestRebalanceDates:
    def test_monthly(self):
        engine = BacktestEngine(top_n=5, rebalance_freq="M")
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="B")
        rb = engine._get_rebalance_dates(dates)

        assert len(rb) > 1     # 至少每月一次
        assert dates[0] in rb   # 首日必调仓
        assert dates[-1] in rb  # 末日必调仓

    def test_weekly(self):
        engine = BacktestEngine(top_n=5, rebalance_freq="W")
        dates = pd.date_range("2024-01-01", "2024-01-31", freq="B")
        rb = engine._get_rebalance_dates(dates)

        assert len(rb) >= 4  # 1月应有4-5周
        assert dates[0] in rb

    def test_daily(self):
        engine = BacktestEngine(top_n=5, rebalance_freq="D")
        dates = pd.date_range("2024-01-02", "2024-01-10", freq="B")
        rb = engine._get_rebalance_dates(dates)

        assert len(rb) == len(dates)


# ============================================================================
# 换手成本测试
# ============================================================================

class TestTurnoverCost:
    def test_no_change(self):
        engine = BacktestEngine(top_n=5)
        old = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=list("ABCDE"))
        new = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=list("ABCDE"))
        cost = engine._calc_turnover_cost(old, new)
        assert cost == 0

    def test_full_change(self):
        engine = BacktestEngine(top_n=5)
        old = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=list("ABCDE"))
        new = pd.Series([0.5, 0.5, 0.0, 0.0, 0.0], index=list("ABCDE"))
        cost = engine._calc_turnover_cost(old, new)
        # 换手 = |0.2-0.5|+|0.2-0.5|+|0.2-0|+|0.2-0|+|0.2-0| = 0.3+0.3+0.2+0.2+0.2 = 1.2
        # 单边 = 1.2/2 = 0.6
        # cost = 0.6 * 0.0031 ≈ 0.00186
        expected = 0.6 * engine.cost.round_trip_cost_ratio()
        assert abs(cost - expected) < 1e-10

    def test_cost_capped(self):
        engine = BacktestEngine(top_n=5)
        old = pd.Series([1.0, 0.0], index=["A", "B"])
        new = pd.Series([0.0, 1.0], index=["A", "B"])
        cost = engine._calc_turnover_cost(old, new)
        # 100%换手, 费率 = 0.31%, cost = 0.0031
        expected = engine.cost.round_trip_cost_ratio()
        assert abs(cost - expected) < 1e-10
        assert cost < 0.05  # 未触上限


# ============================================================================
# 回测全流程测试
# ============================================================================

class TestBacktestFull:
    @pytest.fixture
    def sample_panel(self):
        """构建10只股票、100天、有趋势的模拟面板。"""
        np.random.seed(42)
        stocks = [f"S{i:02d}" for i in range(10)]
        dates = pd.date_range("2024-01-02", periods=100, freq="B")

        data = []
        for i, s in enumerate(stocks):
            # 每只股票有不同趋势
            drift = 0.0002 * (i - 5)  # -0.001 ~ +0.001
            price = 100 * (1 + np.random.randn(100).cumsum() * 0.02 + np.arange(100) * drift)
            for j, d in enumerate(dates):
                data.append({
                    "trade_date": d,
                    "ts_code": s,
                    "close": max(price[j], 1),
                })

        df = pd.DataFrame(data)
        return df.set_index(["trade_date", "ts_code"]).sort_index()

    @pytest.fixture
    def sample_signal(self, sample_panel):
        """构建与面板对齐的信号。"""
        price = sample_panel["close"].unstack("ts_code")
        # 信号 = 短期动量（最近5日收益排名）
        signal = price.pct_change(5).rank(axis=1, pct=True)
        signal.index.name = "trade_date"
        stacked = signal.stack().rename("signal")
        stacked.index.names = ["trade_date", "ts_code"]
        return stacked

    def test_engine_runs(self, sample_panel, sample_signal):
        engine = BacktestEngine(init_cash=1_000_000, top_n=3, rebalance_freq="M")
        result = engine.run(sample_panel, sample_signal, price_col="close")

        assert result.nav is not None
        assert result.daily_returns is not None
        assert result.metrics is not None
        assert "sharpe" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "annual_return" in result.metrics

    def test_nav_starts_at_one(self, sample_panel, sample_signal):
        engine = BacktestEngine(init_cash=1_000_000, top_n=3, rebalance_freq="M")
        result = engine.run(sample_panel, sample_signal, price_col="close")
        assert abs(result.nav.iloc[0] - 1.0) < 0.01

    def test_nav_monotonic_no_gaps(self, sample_panel, sample_signal):
        engine = BacktestEngine(init_cash=1_000_000, top_n=3, rebalance_freq="M")
        result = engine.run(sample_panel, sample_signal, price_col="close")
        assert result.nav.notna().all()

    def test_summary_output(self, sample_panel, sample_signal):
        engine = BacktestEngine(init_cash=1_000_000, top_n=3, rebalance_freq="M")
        result = engine.run(sample_panel, sample_signal, price_col="close")
        summary = result.summary()
        assert "回测绩效报告" in summary
        assert "年化收益" in summary
        assert "最大回撤" in summary

    def test_cost_increases_with_frequency(self, sample_panel, sample_signal):
        """月度调仓成本应小于日度调仓。"""
        engine_m = BacktestEngine(top_n=3, rebalance_freq="M")
        engine_d = BacktestEngine(top_n=3, rebalance_freq="D")

        result_m = engine_m.run(sample_panel, sample_signal, price_col="close")
        result_d = engine_d.run(sample_panel, sample_signal, price_col="close")

        assert result_m.metrics["total_cost"] <= result_d.metrics["total_cost"]

    def test_top_n_one(self, sample_panel, sample_signal):
        """Top 1 股票应能运行。"""
        engine = BacktestEngine(top_n=1, rebalance_freq="M")
        result = engine.run(sample_panel, sample_signal, price_col="close")
        assert result.nav is not None
        assert "sharpe" in result.metrics

    def test_empty_data(self):
        engine = BacktestEngine(top_n=5)
        idx = pd.MultiIndex.from_arrays([[], []], names=["trade_date", "ts_code"])
        empty_panel = pd.DataFrame({"close": []}, index=idx)
        empty_signal = pd.Series([], index=idx, dtype=float)
        result = engine.run(empty_panel, empty_signal, price_col="close")
        assert result.nav is None
        assert result.metrics == {}

    def test_single_day(self, sample_panel):
        """单日数据不应崩溃。"""
        single = sample_panel.loc[sample_panel.index.get_level_values(0)[:1]]
        signal = pd.Series(
            [0.5] * 10,
            index=pd.MultiIndex.from_product(
                [single.index.get_level_values(0).unique(), single.index.get_level_values(1).unique()],
                names=["trade_date", "ts_code"],
            ),
        )
        engine = BacktestEngine(top_n=3)
        result = engine.run(single, signal, price_col="close")
        assert result.nav is None
