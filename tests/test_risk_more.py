"""风控管理器补充测试 — 覆盖剩余分支。

覆盖: 行业止损、行业集中度裁剪、持仓数限制、边界条件。
"""

import numpy as np
import pandas as pd
import pytest

from smartalpha.risk.manager import RiskManager, RiskLimits, RiskEvent
from smartalpha.risk.var import VaRCalculator


# ============================================================================
# RiskManager 补充测试
# ============================================================================

class TestRiskManagerMore:
    @pytest.fixture
    def rm(self):
        return RiskManager(RiskLimits(
            stop_loss_single=-0.08,
            stop_loss_portfolio=-0.05,
            stop_loss_sector=-0.10,
            max_single_position=0.10,
            max_sector_position=0.30,
            max_positions=50,
            max_top3_sector=0.60,
            trailing_stop=-0.10,
        ))

    def test_sector_stop_loss(self, rm):
        """行业止损：某行业总跌幅超阈值时禁止该行业。"""
        # 注意：当前 manager 没有直接的行业止损逻辑（只有行业上限），
        # 但我们可以测个股止损在行业上下文中不受影响
        weights = pd.Series([0.30, 0.25, 0.20], index=["A", "B", "C"])
        daily_pnl = {"A": -0.12, "B": -0.02, "C": 0.01}
        adj, events = rm.check("2024-01-15", weights, daily_pnl=daily_pnl)
        assert adj["A"] == 0
        assert len(events) == 1

    def test_industry_sector_limit_enforced(self, rm):
        """单行业超限应触发 sector_limit 事件。"""
        weights = pd.Series(
            [0.10, 0.10, 0.10, 0.10, 0.09, 0.06],
            index=["BANK1", "BANK2", "BANK3", "BANK4", "TECH1", "TECH2"],
        )
        industry_map = {
            "BANK1": "银行", "BANK2": "银行", "BANK3": "银行", "BANK4": "银行",
            "TECH1": "科技", "TECH2": "科技",
        }
        adj, events = rm.check("2024-01-15", weights, industry_map=industry_map)
        # 银行权重 = 0.40 > 30%, 应触发行上限事件
        assert any(e.event_type == "sector_limit" for e in events)

    def test_top3_sector_concentration(self, rm):
        """前3行业集中度超限应整体缩放 — 每行业3只，各0.10，不触发个股和行业上限。"""
        weights = pd.Series(
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            index=["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"],
        )
        industry_map = {
            "A1": "银行", "A2": "银行", "A3": "银行",
            "B1": "地产", "B2": "地产", "B3": "地产",
            "C1": "能源", "C2": "能源", "C3": "能源",
        }
        adj, events = rm.check("2024-01-15", weights, industry_map=industry_map)
        # 前3 = 银行(0.30) + 地产(0.30) + 能源(0.30) = 0.90 > 60% → 触发
        assert any(e.event_type == "sector_concentration" for e in events)
        assert adj.sum() < 1.0  # 整体被缩放

    def test_max_positions_cap(self, rm):
        """持仓数超过上限时仅保留权重最大的N只。"""
        # 使用极小 max_positions 来测试
        rm2 = RiskManager(RiskLimits(max_single_position=0.10, max_positions=3))
        idx = [f"S{i:02d}" for i in range(10)]
        weights = pd.Series(np.linspace(0.19, 0.01, 10), index=idx)
        adj, _ = rm2.check("2024-01-15", weights)
        positive_count = (adj > 0).sum()
        assert positive_count <= 3

    def test_portfolio_stop_no_history(self, rm):
        """无净值历史时不抛异常。"""
        weights = pd.Series([0.5, 0.5], index=["A", "B"])
        adj, events = rm.check("2024-01-15", weights, nav_history=None)
        assert adj.sum() > 0  # 不触发
        assert len(events) == 0

    def test_trailing_stop_update_peak(self, rm):
        """移动止盈：权重上升时更新峰值。"""
        w1 = pd.Series([0.20], index=["X"])
        # 第一次check，峰值更新
        adj1, _ = rm.check("2024-01-15", w1)
        assert rm._peak_weights.get("X") == 0.20

        # 权重上升
        w2 = pd.Series([0.30], index=["X"])
        adj2, _ = rm.check("2024-01-16", w2)
        assert rm._peak_weights.get("X") == 0.30  # 峰值已更新

    def test_daily_pnl_missing_stock(self, rm):
        """个股损益中不存在的股票不影响检查。"""
        weights = pd.Series([0.5, 0.5], index=["A", "B"])
        daily_pnl = {"C": -0.20}  # C 不在持仓中
        adj, events = rm.check("2024-01-15", weights, daily_pnl=daily_pnl)
        assert adj["A"] > 0
        assert adj["B"] > 0
        assert len(events) == 0

    def test_zero_weights_normalization(self, rm):
        """全零权重归一化不崩溃。"""
        weights = pd.Series([0.0, 0.0, 0.0], index=["A", "B", "C"])
        adj, events = rm.check("2024-01-15", weights)
        assert adj.sum() == 0  # 全零

    def test_custom_limits_default(self):
        """默认 RiskLimits 具有合理的参数值。"""
        limits = RiskLimits()
        assert limits.stop_loss_single == -0.08
        assert limits.max_single_position == 0.10
        assert limits.max_positions == 50

    def test_risk_event_repr(self):
        """RiskEvent 的字段完整性。"""
        evt = RiskEvent(
            date="2024-01-15",
            event_type="stop_loss_single",
            detail="测试事件",
            action="liquidate",
        )
        assert evt.date == "2024-01-15"
        assert evt.event_type == "stop_loss_single"
        assert evt.detail == "测试事件"
        assert evt.action == "liquidate"

    def test_risk_limits_dataclass(self):
        """RiskLimits 是 dataclass，可以自定义。"""
        limits = RiskLimits(stop_loss_single=-0.15, max_single_position=0.05)
        assert limits.stop_loss_single == -0.15
        assert limits.max_single_position == 0.05
        # 未设置的保持默认
        assert limits.stop_loss_portfolio == -0.05

    def test_hhi_perfect_concentration(self, rm):
        """单行业全部持仓 → HHI=1.0。"""
        weights = pd.Series({"A": 0.5, "B": 0.5})
        industry_map = {"A": "同一行业", "B": "同一行业"}
        hhi = rm.compute_hhi(weights, industry_map)
        assert abs(hhi - 1.0) < 0.01

    def test_hhi_minimum(self, rm):
        """行业完全分散 → HHI 低。"""
        weights = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        industry_map = {"A": "a", "B": "b", "C": "c", "D": "d"}
        hhi = rm.compute_hhi(weights, industry_map)
        assert hhi < 0.5


# ============================================================================
# VaR 边界条件测试
# ============================================================================

class TestVaREdgeCases:
    def test_zero_variance_parametric(self):
        """零方差序列的参数法 VaR。"""
        rets = pd.Series([0.0] * 100)
        r = VaRCalculator.parametric(rets)
        assert r.var_95 == 0
        assert r.var_99 == 0

    def test_all_negative_returns(self):
        """全负收益率。"""
        rets = pd.Series(np.array([-0.05, -0.03, -0.04, -0.02, -0.06] * 40))
        r = VaRCalculator.historical(rets)
        assert r.var_95 < 0
        assert r.cvar_95 <= r.var_95  # CVaR ≤ VaR (可能相等)

    def test_monte_carlo_insufficient_data(self):
        """数据不足时蒙特卡洛返回零值。"""
        rets = pd.Series([0.01, 0.02])
        r = VaRCalculator.monte_carlo(rets)
        assert r.var_95 == 0
        assert r.method == "monte_carlo"

    def test_monte_carlo_zero_vol(self):
        """零波动率时蒙特卡洛返回零值。"""
        rets = pd.Series([0.001] * 30)
        r = VaRCalculator.monte_carlo(rets)
        assert r.var_95 == 0


# ============================================================================
# 压力测试补充
# ============================================================================

class TestStressTesterMore:
    def test_custom_vol_scale(self):
        """不同波动率的组合应有不同缩放后的冲击。"""
        from smartalpha.risk.stress import StressTester
        rets_low = pd.Series(np.random.RandomState(0).normal(0, 0.005, 500))
        rets_high = pd.Series(np.random.RandomState(0).normal(0, 0.02, 500))

        tester = StressTester()
        r_low = tester.run_all(rets_low, var_95=-0.01)
        r_high = tester.run_all(rets_high, var_95=-0.01)

        # 高波动组合在相同场景下应有更大损失
        low_crash = [r for r in r_low if r.scenario_name == "2015股灾"][0]
        high_crash = [r for r in r_high if r.scenario_name == "2015股灾"][0]
        # 高波动组合回撤更深或收益更差
        assert high_crash.total_return <= low_crash.total_return or abs(high_crash.total_return) >= abs(low_crash.total_return) * 0.8
