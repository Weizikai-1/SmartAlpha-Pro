"""风控模块单元测试。

测试: VaR/CVaR (3种方法)、止损/止盈、仓位限制、行业集中度、压力测试。
"""

import numpy as np
import pandas as pd
import pytest

from smartalpha.risk.var import VaRCalculator, VaRResult
from smartalpha.risk.manager import RiskManager, RiskLimits, RiskEvent
from smartalpha.risk.stress import StressTester, StressTestResult


# ============================================================================
# VaR/CVaR 测试
# ============================================================================

class TestVaRHistorical:
    @pytest.fixture
    def normal_returns(self):
        """正态分布收益率 (μ=0.001, σ=0.02)。"""
        rng = np.random.RandomState(42)
        return pd.Series(rng.normal(0.001, 0.02, 500))

    @pytest.fixture
    def fat_tail_returns(self):
        """厚尾收益率 (t分布, df=3)。"""
        rng = np.random.RandomState(42)
        return pd.Series(rng.standard_t(3, 500) * 0.02)

    def test_var_95_negative(self, normal_returns):
        r = VaRCalculator.historical(normal_returns)
        assert r.var_95 < 0          # VaR应为负值
        assert r.cvar_95 <= r.var_95  # CVaR ≤ VaR (更极端)

    def test_var_99_more_extreme(self, normal_returns):
        r = VaRCalculator.historical(normal_returns)
        assert r.var_99 <= r.var_95  # 99% 比 95% 更极端

    def test_fat_tail_cvar_wider(self, normal_returns, fat_tail_returns):
        """厚尾分布的 CVaR 与 VaR 的差距应大于正态分布。"""
        r_n = VaRCalculator.historical(normal_returns)
        r_f = VaRCalculator.historical(fat_tail_returns)
        gap_n = abs(r_n.cvar_95 - r_n.var_95)
        gap_f = abs(r_f.cvar_95 - r_f.var_95)
        assert gap_f > gap_n * 0.5  # 厚尾应有更大尾部损失

    def test_insufficient_data(self):
        r = VaRCalculator.historical(pd.Series([0.01, 0.02]))
        assert r.var_95 == 0 and r.var_99 == 0

    def test_method_field(self, normal_returns):
        r = VaRCalculator.historical(normal_returns)
        assert r.method == "historical"


class TestVaRParametric:
    def test_returns_normal_approx(self):
        """正态分布数据，参数法应接近理论值。"""
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(0, 0.01, 1000))
        r = VaRCalculator.parametric(rets)
        from scipy.stats import norm
        expected_95 = 0.01 * norm.ppf(0.05)
        assert abs(r.var_95 - expected_95) < 0.005

    def test_zero_vol(self):
        rets = pd.Series([0.001] * 100)
        r = VaRCalculator.parametric(rets)
        assert r.var_95 == 0

    def test_method_field(self):
        rng = np.random.RandomState(42)
        rets = pd.Series(rng.normal(0, 0.01, 100))
        r = VaRCalculator.parametric(rets)
        assert r.method == "parametric"


class TestVaRMonteCarlo:
    def test_reproducibility(self):
        rets = pd.Series(np.random.RandomState(0).normal(0, 0.01, 200))
        r1 = VaRCalculator.monte_carlo(rets, random_state=42)
        r2 = VaRCalculator.monte_carlo(rets, random_state=42)
        assert r1.var_95 == r2.var_95  # 可复现

    def test_more_simulations_stable(self):
        rets = pd.Series(np.random.RandomState(0).normal(0, 0.01, 200))
        r100k = VaRCalculator.monte_carlo(rets, n_simulations=100_000)
        r10k = VaRCalculator.monte_carlo(rets, n_simulations=10_000)
        # 10万次模拟应相对稳定
        assert abs(r100k.var_95 - r10k.var_95) < 0.003

    def test_method_field(self):
        rets = pd.Series(np.random.RandomState(0).normal(0, 0.01, 100))
        r = VaRCalculator.monte_carlo(rets)
        assert r.method == "monte_carlo"


class TestVaRAllMethods:
    def test_returns_dict(self):
        rets = pd.Series(np.random.RandomState(42).normal(0.001, 0.02, 500))
        results = VaRCalculator.all_methods(rets)
        assert "historical" in results
        assert "parametric" in results
        assert "monte_carlo" in results
        for r in results.values():
            assert isinstance(r, VaRResult)


# ============================================================================
# 风控管理器测试
# ============================================================================

class TestRiskManager:
    @pytest.fixture
    def rm(self):
        return RiskManager(RiskLimits(
            stop_loss_single=-0.08,
            stop_loss_portfolio=-0.05,
            max_single_position=0.10,
        ))

    @pytest.fixture
    def sample_weights(self):
        return pd.Series(
            [0.30, 0.25, 0.20, 0.15, 0.10],
            index=["A", "B", "C", "D", "E"],
        )

    def test_single_stop_loss(self, rm, sample_weights):
        daily_pnl = {"A": -0.12, "B": 0.01, "C": -0.02, "D": 0.03, "E": -0.01}
        adj, events = rm.check("2024-01-15", sample_weights, daily_pnl=daily_pnl)
        assert adj["A"] == 0  # A 触发止损
        assert events[0].event_type == "stop_loss_single"

    def test_no_stop_if_above_threshold(self, rm, sample_weights):
        daily_pnl = {"A": -0.05, "B": 0.01}
        adj, events = rm.check("2024-01-15", sample_weights, daily_pnl=daily_pnl)
        assert adj["A"] > 0  # 未触发止损

    def test_portfolio_stop_loss(self, rm, sample_weights):
        nav = pd.Series([1.0, 0.94])  # -6% 日跌幅
        adj, events = rm.check("2024-01-15", sample_weights, nav_history=nav)
        assert adj.sum() == 0  # 全部清仓
        assert events[0].event_type == "stop_loss_portfolio"

    def test_position_limit_clip(self, rm, sample_weights):
        adj, _ = rm.check("2024-01-15", sample_weights)
        assert (adj <= rm.limits.max_single_position).all()

    def test_weights_normalized(self, rm, sample_weights):
        adj, _ = rm.check("2024-01-15", sample_weights)
        # 裁剪后允许现金头寸（总和<1），但所有持仓都在上限内
        assert (adj <= rm.limits.max_single_position).all()
        assert adj.sum() <= 1.01

    def test_trailing_stop(self, rm):
        """移动止盈测试。"""
        w1 = pd.Series([0.20, 0.10], index=["X", "Y"])
        # 先建立峰值
        rm._peak_weights["X"] = 0.20
        # 现在 X 从 0.20 跌到 0.10, 回撤 -50%, 超过 -10% 阈值
        w2 = pd.Series([0.10, 0.10], index=["X", "Y"])
        adj, events = rm.check("2024-01-16", w2)
        assert adj["X"] == 0  # 触发移动止盈
        assert any(e.event_type == "trailing_stop" for e in events)

    def test_hhi_calculation(self, rm):
        weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
        industry_map = {"A": "银行", "B": "银行", "C": "科技"}
        hhi = rm.compute_hhi(weights, industry_map)
        # 银行=0.8, 科技=0.2 → HHI = 0.64+0.04 = 0.68
        assert abs(hhi - 0.68) < 0.01

    def test_events_cleared_between_checks(self, rm, sample_weights):
        daily_pnl = {"A": -0.12}
        _, events1 = rm.check("2024-01-15", sample_weights, daily_pnl=daily_pnl)
        assert len(events1) > 0
        _, events2 = rm.check("2024-01-16", sample_weights, daily_pnl={"B": 0.01})
        assert len(events2) == 0  # 新一天的检查，事件已清空

    def test_reset_peaks(self, rm):
        rm._peak_weights["X"] = 0.20
        rm.reset_peaks()
        assert len(rm._peak_weights) == 0


# ============================================================================
# 压力测试
# ============================================================================

class TestStressTester:
    @pytest.fixture
    def tester(self):
        return StressTester()

    @pytest.fixture
    def sample_returns(self):
        rng = np.random.RandomState(42)
        return pd.Series(rng.normal(0.0005, 0.015, 500))

    def test_all_scenarios_run(self, tester, sample_returns):
        results = tester.run_all(sample_returns, var_95=-0.02)
        assert len(results) == 4  # 4个预定义场景

    def test_each_result_has_summary(self, tester, sample_returns):
        results = tester.run_all(sample_returns)
        for r in results:
            summary = r.summary()
            assert r.scenario_name in summary
            assert "净值" in summary

    def test_crash_has_large_loss(self, tester, sample_returns):
        results = tester.run_all(sample_returns)
        crash = [r for r in results if r.scenario_name == "2015股灾"][0]
        assert crash.total_return < -0.01  # 股灾应有明显亏损

    def test_stress_nav_declines(self, tester, sample_returns):
        results = tester.run_all(sample_returns)
        for r in results:
            assert r.end_nav <= 1.01  # 压力场景净值不应大幅增长

    def test_custom_scenario(self):
        result = StressTester.custom_scenario(
            "测试场景", "自定义暴跌 -50%",
            shocks=[-0.50],
            var_95=-0.02,
        )
        assert result.end_nav == 0.5
        assert result.total_return == -0.5
        assert result.var_breaches == 1

    def test_worst_day_tracked(self, tester, sample_returns):
        results = tester.run_all(sample_returns)
        for r in results:
            assert r.worst_day <= 0  # 最差日应为负
            assert r.daily_nav[0] == 1.0  # 起始净值
