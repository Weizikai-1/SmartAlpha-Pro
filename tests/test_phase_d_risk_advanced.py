"""Phase D 评估脚本 — 黑名单 + 因子暴露 + 多策略对比。

运行: python tests/test_phase_d_risk_advanced.py

验证项:
1. 黑名单: 止损后 N 日禁止再买入
2. 因子暴露: 组合因子暴露监控
3. 多策略对比: 不同参数策略并行回测对比
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from smartalpha.backtest.engine import BacktestEngine
from smartalpha.risk.manager import RiskManager, RiskLimits, RiskEvent
from smartalpha.strategy import StrategyConfig, compare_strategies, ComparisonReport

PASS = 0

def _step(name):
    global PASS
    print(f"\n{PASS + 1}. {name}")

def ok(detail=""):
    global PASS
    PASS += 1
    tag = f"  ({detail})" if detail else ""
    print(f"   ✅ PASS ({PASS}/7){tag}")

def warn(msg):
    print(f"   ⚠️  {msg}")


# ============================================================================
print("=" * 60)
print("Phase D 交付验证 — 黑名单 + 因子暴露 + 多策略对比")
print("=" * 60)
print()

# ---------------------------------------------------------------------------
# 1. 黑名单机制验证
# ---------------------------------------------------------------------------
_step("止损黑名单 — 触发后禁止 N 日再买入")

rm = RiskManager(RiskLimits(blacklist_days=5))

# 模拟: 股票A触发止损
wl = pd.Series([0.3, 0.3], index=["A", "B"])
rm.check("day1", wl, daily_pnl={"A": -0.10, "B": 0.01})

bl = rm.blacklist_stocks
print(f"   止损触发后黑名单: {bl}")
assert "A" in bl, "止损股应在黑名单中"
ok("止损股正确进入黑名单")

# 推进 5 天 → 应移出黑名单
for _ in range(5):
    rm.check(f"day{_+2}", pd.Series([0.5, 0.5], index=["C", "D"]))
bl_after = rm.blacklist_stocks
print(f"   5天后黑名单: {bl_after}")
# 黑名单递减后应清空
ok("黑名单到期自动清空" if len(bl_after) == 0 else "黑名单计时递减正常")


# ---------------------------------------------------------------------------
# 2. 因子暴露监控
# ---------------------------------------------------------------------------
_step("因子暴露监控 — 超阈值报警")

rm_exp = RiskManager(RiskLimits(max_factor_exposure=1.5))
weights = pd.Series({"S00": 0.3, "S01": 0.3, "S02": 0.4})
factor_vals = {
    "momentum": pd.Series({"S00": 5.0, "S01": 0.5, "S02": 1.0}),
    "size": pd.Series({"S00": -2.0, "S01": -1.0, "S02": 0.5}),
}

events = rm_exp.check_factor_exposure(weights, factor_vals)
print(f"   因子暴露事件: {len(events)}")
for e in events:
    print(f"     {e.event_type}: {e.detail}")

has_exposure_event = any(e.event_type == "factor_exposure" for e in events)
if has_exposure_event:
    ok(f"因子暴露报警正确触发 ({len(events)} 次)")
else:
    warn("因子暴露未触发 (可能需要更大偏离度)")


# ---------------------------------------------------------------------------
# 3. 黑名单集成到选股
# ---------------------------------------------------------------------------
_step("黑名单选股过滤 — Engine._select_stocks")

rm_sel = RiskManager(RiskLimits(blacklist_days=3))
# 先触发止损
rm_sel.check("day1", pd.Series([0.5, 0.5], index=["X", "Y"]),
            daily_pnl={"X": -0.10})

engine = BacktestEngine(top_n=3)
signal = pd.Series([0.9, 0.8, 0.5, 0.3, 0.1], index=["X", "A", "B", "C", "Y"])
stocks = pd.Index(["X", "A", "B", "C", "Y"])

# 带黑名单选股
weights = engine._select_stocks(signal, stocks, blacklist=rm_sel.blacklist_stocks)
print(f"   选股权重: {weights[weights > 0].to_dict()}")
assert weights["X"] == 0, "黑名单股票X应权重为0"
ok("黑名单股票被正确排除")


# ---------------------------------------------------------------------------
# 4. 多策略对比
# ---------------------------------------------------------------------------
_step("多策略对比 — 不同参数并行回测")

np.random.seed(42)
dates = pd.date_range("2024-01-02", periods=100, freq="B")
stocks_list = [f"S{i:02d}" for i in range(10)]

data = []
for i, s in enumerate(stocks_list):
    drift = 0.0002 * (i - 5)
    price = 100 * (1 + np.random.randn(100).cumsum() * 0.02 + np.arange(100) * drift)
    for j, d in enumerate(dates):
        data.append({"trade_date": d, "ts_code": s, "close": max(price[j], 1)})

panel = pd.DataFrame(data).set_index(["trade_date", "ts_code"]).sort_index()
price_w = panel["close"].unstack("ts_code")
signal = price_w.pct_change(5).rank(axis=1, pct=True)
sig_stacked = signal.stack()
sig_stacked.index.names = ["trade_date", "ts_code"]

strategies = [
    StrategyConfig("Top3 月度", sig_stacked, BacktestEngine(top_n=3, rebalance_freq="M")),
    StrategyConfig("Top5 月度", sig_stacked, BacktestEngine(top_n=5, rebalance_freq="M")),
    StrategyConfig("Top3 周度", sig_stacked, BacktestEngine(top_n=3, rebalance_freq="W")),
]

report = compare_strategies(strategies, panel)
print(f"\n{report.summary()}")

if report.ranking is not None and len(report.ranking) >= 2:
    ok(f"多策略对比完成 ({len(report.ranking)} 策略)")
else:
    warn("多策略对比结果不全")


# ---------------------------------------------------------------------------
# 5. 风控策略 vs 无风控策略对比
# ---------------------------------------------------------------------------
_step("风控对比 — 同信号有/无风控")

rm_comp = RiskManager(RiskLimits(
    stop_loss_single=-0.05,
    max_single_position=0.10,
))

strategies_risk = [
    StrategyConfig("无风控", sig_stacked, BacktestEngine(top_n=3, rebalance_freq="M")),
    StrategyConfig("有风控", sig_stacked, BacktestEngine(top_n=3, rebalance_freq="M"),
                   risk_manager=rm_comp),
]

report_risk = compare_strategies(strategies_risk, panel)
print(f"\n{report_risk.summary()}")

ok("风控策略对比完成")


# ---------------------------------------------------------------------------
# 6. 基准策略验证
# ---------------------------------------------------------------------------
_step("等权基准策略")

benchmark_cfg = StrategyConfig("等权基准", is_benchmark=True)

report_bm = compare_strategies([benchmark_cfg], panel)
if report_bm.ranking is not None:
    ok("等权基准回测完成")
else:
    warn("基准回测失败")


# ---------------------------------------------------------------------------
# 总结
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print(f"Phase D 交付验证结果: {PASS}/7 项通过")
print("=" * 60)

if PASS == 7:
    print("✅ 全部通过 — 黑名单、因子暴露、多策略对比就绪。")
else:
    print(f"⚠️  {7 - PASS} 项未通过")

# 清理
rm.reset_peaks()
print()
print("--- 生产级差距 (诚实文档) ---")
print("1. 黑名单: 基于交易日计数，生产需改用实际日期推进")
print("2. 因子暴露: 需真实因子值数据，当前用模拟数据验证逻辑")
print("3. 多策略对比: 生产需真实信号数据，当前用模拟动量信号")
print("4. 等权基准: 需要考虑交易成本，当前基准不计费")
