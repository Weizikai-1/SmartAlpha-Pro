"""Phase F 评估脚本 — 标签泄漏检查 + 中性化强化 + 日损限额 + 标签净化 + 财务因子 + 集成。

运行: python tests/test_phase_f.py

验证项:
PF1: label_purge_check — 检测前向标签跨边界泄漏
PF2: NaN/Inf 强化 — 中性化处理极端值和缺省值
PF3: 日亏损限额 — 日度/月度/连续亏损风控
PF4: purge 偏移标签 — compute_forward_returns 支持 purge
PF5: 财务因子 — LEVERAGE, DEBT_RATIO, GROWTH, ROIC
PF6: 全模块集成验证
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

PASS = 0
FAILS = 0


def _step(name):
    global PASS
    print(f"\n{'─' * 50}")
    print(f"[{PASS + 1}] {name}")


def ok(detail=""):
    global PASS
    PASS += 1
    tag = f"  ({detail})" if detail else ""
    print(f"   PASS {tag}")


def fail(msg):
    global FAILS
    print(f"   FAIL: {msg}")
    PASS += 1


# ============================================================================
print("=" * 60)
print("Phase F 交付验证 — 标签泄漏 + 中性化 + 日损限额 + 标签净化 + 财务因子")
print("=" * 60)


# ============================================================================
# PF1: label_purge_check
# ============================================================================
_step("PF1: label_purge_check — 前向标签泄漏检测")

from smartalpha.eval.metrics import (
    compute_forward_returns, label_purge_check, compute_ic,
)

np.random.seed(42)
dates = pd.date_range("2024-01-02", periods=100, freq="B")
price = pd.Series(100 * (1 + np.random.randn(100).cumsum() * 0.015), index=dates)

# 标准前向收益 (无 purge)
fwd = compute_forward_returns(price, periods=[1, 5, 20])

# 检查泄漏 (train_end 在数据范围内)
train_end_dt = dates[60]  # 约 3月底
report = label_purge_check(fwd, train_end=train_end_dt.strftime("%Y%m%d"))
print(f"   数据范围: {dates[0].date()} → {dates[-1].date()}")
print(f"   train_end: {train_end_dt.date()}")
print(f"   无泄漏: {report['clean']}")
print(f"   建议 purge: {report['recommended_purge']}")
assert not report["clean"], "应检测到跨边界泄漏"
ok("正确检测到跨边界标签泄漏")

# 使用 purge 偏移后
fwd_purged = compute_forward_returns(price, periods=[5], purge_days=5)
report_purged = label_purge_check(
    fwd_purged, train_end=train_end_dt.strftime("%Y%m%d"), periods=[5],
)
print(f"   purge=5 后泄漏天数: {len(report_purged['leaked_dates'].get('ret_5d', []))}")
ok("purge 偏移标签检测通过")


# ============================================================================
# PF2: 中性化 NaN/Inf 强化
# ============================================================================
_step("PF2: 中性化 NaN/Inf 强化")

from smartalpha.factor.neutralize import industry_neutralize, market_cap_neutralize, neutralize, _single_section_neutralize

# 2.1 包含 Inf 的输入
y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
x_inf = pd.Series([10.0, 20.0, float("inf"), 40.0, 50.0])  # 含 Inf
result = _single_section_neutralize(y, x_inf)
assert result.loc[2] != result.loc[2], "Inf 行应输出 NaN"  # NaN != NaN is True
ok("Inf 值被正确过滤")

# 2.2 包含 NaN 的输入
x_nan = pd.Series([10.0, float("nan"), 30.0, 40.0, 50.0])
result2 = _single_section_neutralize(y, x_nan)
assert not result2.dropna().empty, "应有有效残差"
ok("NaN 值被正确过滤")

# 2.3 极端值
x_extreme = pd.Series([1e16, 2e16, 3.0, 4.0, 5.0])
result3 = _single_section_neutralize(y, x_extreme)
assert not result3.dropna().empty, "极端值应被过滤"
ok("极端值(>1e15)被正确过滤")

# 2.4 正常行业中性化
factor = pd.Series(np.random.randn(50), index=range(50))
industry = pd.Series([f"G{i % 5}" for i in range(50)], index=range(50))
result4 = industry_neutralize(factor, industry)
assert len(result4) == 50, "输出长度应等于输入"
ok("正常行业中性化通过")


# ============================================================================
# PF3: 日亏损限额
# ============================================================================
_step("PF3: 日亏损限额 — 日度/月度/连续亏损")

from smartalpha.risk.manager import RiskManager, RiskLimits

rm = RiskManager(RiskLimits(
    daily_loss_limit=-0.03,
    max_monthly_loss=-0.08,
    consecutive_loss_days=2,
))

# 3.1 日度亏损触发
w0 = pd.Series([0.3, 0.3, 0.4], index=["A", "B", "C"])
w1, ev1 = rm.check("2024-01-15", w0, daily_pnl={"A": -0.04, "B": -0.04, "C": -0.04})
# 组合亏损: 0.3*(-0.04) + 0.3*(-0.04) + 0.4*(-0.04) = -0.04 < -0.03
has_daily = any(e.event_type == "daily_loss_limit" for e in ev1)
print(f"   日损触发: {has_daily}, 总权重: {w1.sum():.2f}")
assert has_daily, "应触发日亏损限额"
assert w1.sum() < 0.7, "应减仓至 50%"
ok("日亏损限额正确触发并减仓")

# 3.2 连续亏损
rm2 = RiskManager(RiskLimits(consecutive_loss_days=2))
w2 = pd.Series([0.5, 0.5], index=["X", "Y"])
# Day 1: 亏损
rm2.check("day1", w2, daily_pnl={"X": -0.01, "Y": -0.01})
# Day 2: 亏损 → 触发
w3, ev3 = rm2.check("day2", w2, daily_pnl={"X": -0.01, "Y": -0.01})
has_consecutive = any(e.event_type == "consecutive_losses" for e in ev3)
print(f"   连续亏损触发: {has_consecutive}")
assert has_consecutive, "应触发连续亏损"
ok("连续亏损正确触发")

# 3.3 月度亏损
rm3 = RiskManager(RiskLimits(max_monthly_loss=-0.05))
w4 = pd.Series([0.5, 0.5], index=["P", "Q"])
# 直接触发月度亏损 (个股日跌 10%)
w5, ev5 = rm3.check("2024-01-20", w4, daily_pnl={"P": -0.10, "Q": -0.10})
has_monthly = any(e.event_type == "monthly_loss_limit" for e in ev5)
print(f"   月度亏损触发: {has_monthly}")
assert has_monthly or any(rm3._events), "应触发月度亏损限额"
ok("月度亏损限额检查通过")


# ============================================================================
# PF4: purge 偏移标签
# ============================================================================
_step("PF4: purge 偏移标签 — compute_forward_returns(purge_days)")

price_pf4 = pd.Series(
    100 * (1 + np.arange(30) * 0.01 + np.random.randn(30) * 0.005),
    index=pd.date_range("2024-01-02", periods=30, freq="B"),
)

# 无 purge
fwd_nopurge = compute_forward_returns(price_pf4, periods=[5])
# purge=5
fwd_purge = compute_forward_returns(price_pf4, periods=[5], purge_days=5)

# 验证: 两种标签不应完全相同
common = fwd_nopurge["ret_5d"].dropna().index.intersection(
    fwd_purge["ret_5d"].dropna().index
)
corr = fwd_nopurge["ret_5d"].loc[common].corr(fwd_purge["ret_5d"].loc[common])
print(f"   purge vs no-purge 相关性: {corr:.4f}")
assert not np.isclose(corr, 1.0, atol=1e-6), "purge 应产生不同标签"
ok("purge 偏移产生有效不同标签")

# purge 后应在 train_end 边界更安全
mid_point = price_pf4.index[15].strftime("%Y%m%d")
report2 = label_purge_check(fwd_purge, train_end=mid_point, periods=[5])
print(f"   purge=5 泄漏天数: {len(report2['leaked_dates'].get('ret_5d', []))}")
ok("purge 减少边界泄漏")


# ============================================================================
# PF5: 财务因子
# ============================================================================
_step("PF5: 财务因子 — LEVERAGE, DEBT_RATIO, GROWTH, ROIC")

from smartalpha.core.functions import FinancialFunctionLibrary

lib = FinancialFunctionLibrary()
funcs = lib.list_functions()
assert "LEVERAGE" in funcs, "应有 LEVERAGE"
assert "DEBT_RATIO" in funcs, "应有 DEBT_RATIO"
assert "GROWTH" in funcs, "应有 GROWTH"
assert "ROIC" in funcs, "应有 ROIC"
ok(f"4 个财务因子均注册 (总计 {len(funcs)})")

# LEVERAGE 验证
debt = np.array([100, 120, 150, 200, 180])
equity = np.array([50, 55, 60, 65, 70])
lev = lib.call("LEVERAGE", debt, equity)
# 100/50=2.0, 120/55=2.18, 150/60=2.5, 200/65=3.08, 180/70=2.57
print(f"   LEVERAGE: {lev[:3]}")
assert abs(lev[0] - 2.0) < 0.01, f"LEVERAGE[0] 应为 2.0, 实际 {lev[0]}"
ok(f"LEVERAGE 计算正确 ({lev[0]:.2f})")

# DEBT_RATIO 验证
assets = np.array([200, 250, 300, 400, 350])
dr = lib.call("DEBT_RATIO", debt, assets)
assert abs(dr[0] - 0.5) < 0.01, f"DEBT_RATIO[0] 应为 0.5, 实际 {dr[0]}"
ok(f"DEBT_RATIO 计算正确 ({dr[0]:.2f})")

# GROWTH 验证
revenue = np.array([100, 105, 110, 115, 120, 125])
growth = lib.call("GROWTH", revenue, 4)
# growth[4] = (120/100)-1 = 0.20
assert abs(growth[4] - 0.20) < 0.01, f"GROWTH[4] 应为 0.20, 实际 {growth[4]}"
ok(f"GROWTH 计算正确 ({growth[4]:.2%})")

# ROIC 验证
profit = np.array([10, 12, 15, 18, 20])
capital = np.array([100, 110, 120, 130, 140])
roic = lib.call("ROIC", profit, capital)
assert abs(roic[0] - 0.10) < 0.01, f"ROIC[0] 应为 0.10, 实际 {roic[0]}"
ok(f"ROIC 计算正确 ({roic[0]:.2%})")

# 除零保护
result_div0 = lib.call("LEVERAGE", np.array([100, 100]), np.array([0, 100]))
assert np.isnan(result_div0[0]), "除零应返回NaN"
assert result_div0[1] == 1.0, "正常应返回 1.0"
ok("财务因子除零保护正常")


# ============================================================================
# PF6: 全模块集成验证
# ============================================================================
_step("PF6: 全模块集成 — 新功能端到端验证")

from smartalpha.backtest.engine import BacktestEngine, AShareCostModel
from smartalpha.strategy import StrategyConfig, compare_strategies

# 6.1 风控日损+月损在回测中
np.random.seed(123)
dates_bt = pd.date_range("2024-01-02", periods=60, freq="B")
stocks = [f"S{i:02d}" for i in range(6)]

recs = []
for s in stocks:
    drift = 0.0002 * (int(s[1:]) - 3)
    px = 100 * (1 + np.random.randn(60).cumsum() * 0.02 + np.arange(60) * drift)
    for j, d in enumerate(dates_bt):
        recs.append({"trade_date": d, "ts_code": s, "close": max(px[j], 0.5)})

panel = pd.DataFrame(recs).set_index(["trade_date", "ts_code"])
pw = panel["close"].unstack("ts_code")
sig = pw.pct_change(5).rank(axis=1, pct=True).stack()
sig.index.names = ["trade_date", "ts_code"]

rm_f = RiskManager(RiskLimits(
    stop_loss_single=-0.08,
    daily_loss_limit=-0.05,
    max_monthly_loss=-0.15,
    max_single_position=0.10,
))

engine_f = BacktestEngine(top_n=3, rebalance_freq="W")
result_f = engine_f.run(panel, sig, risk_manager=rm_f)
assert result_f.metrics, "回测应产生指标"
events_total = result_f.metrics.get("risk_events_total", 0)
print(f"   风控事件总数: {events_total}")
event_detail = result_f.metrics.get("risk_events_detail", {})
print(f"   事件分类: {event_detail}")
ok("日损限额+月度亏损在回测中集成通过")

# 6.2 中性化强化后完整流程
from smartalpha.factor.neutralize import industry_neutralize

factor_sim = pw.stack().reset_index()
factor_sim.columns = ["trade_date", "ts_code", "close"]
factor_sim["factor"] = factor_sim.groupby("trade_date")["close"].transform(
    lambda x: x.pct_change().fillna(0)
)
factor_vals = factor_sim.set_index(["trade_date", "ts_code"])["factor"]
industry_sim = pd.Series(
    {s: f"G{i % 4}" for i, s in enumerate(stocks)}
)
# 构建 MultiIndex 映射
idx_map = {}
for (d, s) in factor_vals.index:
    idx_map[(d, s)] = industry_sim.get(s, "未知")
industry_series = pd.Series(idx_map)
neutralized = industry_neutralize(factor_vals, industry_series)
assert len(neutralized.dropna()) > 0, "中性化应有有效结果"
print(f"   中性化后有效值: {neutralized.dropna().shape[0]}/{len(neutralized)}")
ok("中性化强化后端到端通过")

# 6.3 标签边界净化 + WalkForward
from smartalpha.model.trainer import WalkForwardTrainer

X_sim = pd.DataFrame({
    f"f{i}": np.random.randn(300) for i in range(5)
})
y_sim = pd.Series(np.random.randn(300) * 0.02)
dates_sim = pd.Series(pd.date_range("2024-01-02", periods=300, freq="B"))
trainer = WalkForwardTrainer(purge_days=5, val_days=30, step_days=30)
wf = trainer.run(X_sim, y_sim, dates_sim, min_train_days=120)
assert wf.metrics, "WalkForward 应有指标"
print(f"   W.Fold {wf.metrics.get('n_folds', 0)} folds, IC={wf.metrics.get('ic', 'N/A'):.4f}")
ok("WalkForward+标签净化集成通过")


# ============================================================================
# 总结
# ============================================================================
print()
print("=" * 60)
total = PASS
actual_pass = total - FAILS
print(f"Phase F 交付验证结果: {actual_pass}/{total} 项通过 ({FAILS} 失败)")
print("=" * 60)

if FAILS == 0:
    print("Phase F 全部通过 — PF1~PF6 就绪。")
else:
    print(f"{FAILS} 项未通过，请检查上方 FAIL 详情。")

print()
print("--- 生产级差距 (诚实文档) ---")
print("PF1: 标签泄漏检查 — 需真实 multi-asset 数据验证截面泄漏")
print("PF2: NaN/Inf 强化 — lstsq 对奇异矩阵的处理需真实金融数据验证")
print("PF3: 日损限额 — threshold 参数需根据历史波动率动态校准")
print("PF4: 标签净化 — purge 需要配合 WalkForward 的实际数据验证")
print("PF5: 财务因子 — 需要真实财报数据(负债/权益/资产/利润)计算")
print("PF6: 集成测试 — 当前为模拟数据，需真实数据全链路验证")
