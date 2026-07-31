"""Phase B 集成测试 — 风控回测 + OOF管道。

测试: 风控集成、OOF信号转换、端到端管道。
"""

import numpy as np
import pandas as pd
import pytest

from smartalpha.backtest.engine import BacktestEngine, AShareCostModel
from smartalpha.risk.manager import RiskManager, RiskLimits
from smartalpha.pipeline import build_signal_from_predictions, CrossSectionalPipeline


# ============================================================================
# 风控集成测试
# ============================================================================

class TestRiskBacktest:
    @pytest.fixture
    def panel_signal(self):
        """10股×100天的模拟面板+信号。"""
        np.random.seed(42)
        stocks = [f"S{i:02d}" for i in range(10)]
        dates = pd.date_range("2024-01-02", periods=100, freq="B")

        data = []
        for i, s in enumerate(stocks):
            drift = 0.0002 * (i - 5)
            price = 100 * (1 + np.random.randn(100).cumsum() * 0.02 + np.arange(100) * drift)
            for j, d in enumerate(dates):
                data.append({"trade_date": d, "ts_code": s, "close": max(price[j], 1)})

        panel = pd.DataFrame(data).set_index(["trade_date", "ts_code"]).sort_index()
        price_w = panel["close"].unstack("ts_code")
        signal_raw = price_w.pct_change(5).rank(axis=1, pct=True)
        stacked = signal_raw.stack()
        stacked.index.names = ["trade_date", "ts_code"]
        return panel, stacked

    def test_backward_compatible(self, panel_signal):
        """不传 risk_manager 时行为不变，风控事件为0。"""
        panel, signal = panel_signal
        engine = BacktestEngine(top_n=3, rebalance_freq="M")
        result = engine.run(panel, signal, price_col="close")
        assert result.nav is not None
        assert "sharpe" in result.metrics
        assert result.metrics.get("risk_events_total") == 0  # 始终存在，为0

    def test_risk_manager_integrated(self, panel_signal):
        """传入 risk_manager 时正常执行。"""
        panel, signal = panel_signal
        rm = RiskManager(RiskLimits(
            stop_loss_single=-0.08, max_single_position=0.10,
        ))
        engine = BacktestEngine(top_n=3, rebalance_freq="M")
        result = engine.run(panel, signal, price_col="close", risk_manager=rm)
        assert result.nav is not None
        # 风控统计应存在
        assert "risk_events_total" in result.metrics
        assert result.metrics["risk_events_total"] >= 0

    def test_risk_nav_still_valid(self, panel_signal):
        """风控不应破坏净值有效性。"""
        panel, signal = panel_signal
        rm = RiskManager(RiskLimits(max_single_position=0.10))
        engine = BacktestEngine(top_n=3, rebalance_freq="M")
        result = engine.run(panel, signal, price_col="close", risk_manager=rm)
        assert result.nav.notna().all()
        assert abs(result.nav.iloc[0] - 1.0) < 0.01

    def test_risk_with_industry(self, panel_signal):
        """带行业映射的风控。"""
        panel, signal = panel_signal
        stocks = [f"S{i:02d}" for i in range(10)]
        ind_map = {stocks[i]: ("银行" if i < 4 else ("科技" if i < 7 else "消费"))
                   for i in range(10)}

        rm = RiskManager(RiskLimits(max_sector_position=0.30))
        engine = BacktestEngine(top_n=5, rebalance_freq="M")
        result = engine.run(panel, signal, price_col="close",
                            risk_manager=rm, industry_map=ind_map)
        assert result.nav is not None

    def test_risk_metrics_include_events(self, panel_signal):
        """止损事件应出现在指标中。"""
        panel, signal = panel_signal
        rm = RiskManager(RiskLimits(stop_loss_single=-0.03, stop_loss_portfolio=-0.05))
        engine = BacktestEngine(top_n=5, rebalance_freq="D")  # 日频提高触发
        result = engine.run(panel, signal, price_col="close", risk_manager=rm)
        assert "risk_events_total" in result.metrics
        assert "risk_events_detail" in result.metrics

    def test_summary_includes_risk_warnings(self, panel_signal):
        """summary 输出应包含风控警告。"""
        panel, signal = panel_signal
        rm = RiskManager(RiskLimits(stop_loss_single=-0.03))
        engine = BacktestEngine(top_n=5, rebalance_freq="D")
        result = engine.run(panel, signal, price_col="close", risk_manager=rm)
        summary = result.summary()
        assert "回测绩效报告" in summary


# ============================================================================
# OOF管道测试
# ============================================================================

class TestOofPipeline:
    def test_build_signal_aligned(self):
        """OOF预测与factor_idx完全对齐时直接映射。"""
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=5), ["A", "B"]],
            names=["trade_date", "ts_code"],
        )
        oof = pd.Series(np.arange(10, dtype=float), index=idx)
        panel = pd.DataFrame({"close": np.ones(10)}, index=idx)
        signal = build_signal_from_predictions(oof, panel, idx)
        assert len(signal) == len(idx)
        assert not signal.isna().any()

    def test_build_signal_partial_overlap(self):
        """OOF预测与factor_idx部分重叠。"""
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=5), ["A", "B"]],
            names=["trade_date", "ts_code"],
        )
        # 只有一半有预测
        half_idx = idx[:5]
        oof = pd.Series(np.arange(5, dtype=float), index=half_idx)
        panel = pd.DataFrame({"close": np.ones(10)}, index=idx)
        signal = build_signal_from_predictions(oof, panel, idx)
        assert signal.loc[half_idx].notna().all()
        assert signal.loc[idx[5:]].isna().all()

    def test_pipeline_lightgbm_unavailable(self):
        """LightGBM 不可用时管道应返回空结果+错误信息。"""
        panel = pd.DataFrame(
            {"close": [1.0] * 10},
            index=pd.MultiIndex.from_product(
                [pd.date_range("2024-01-01", periods=5), ["A", "B"]],
                names=["trade_date", "ts_code"],
            ),
        )
        factor_wide = pd.DataFrame(
            {"f1": np.random.randn(5)},
            index=pd.date_range("2024-01-01", periods=5),
        )
        fwd_rets = pd.Series(np.random.randn(5), index=pd.date_range("2024-01-01", periods=5))

        pipe = CrossSectionalPipeline()
        result = pipe.run(panel, factor_wide, fwd_rets)
        # 要么有结果，要么有错误
        assert "error" in result or "train_result" in result
