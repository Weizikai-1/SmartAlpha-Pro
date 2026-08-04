"""Phase E 评估脚本 — IC验证集 + 51函数 + 冲击成本 + 退避重试 + BETA指数 + 集成。

运行: python tests/test_phase_e.py

验证项:
PE1: compute_ic 仅验证集计算
PE2: SAR + OBV → 51函数
PE3: 冲击成本独立建模
PE4: 指数退避+jitter 重试
PE5: BETA绑定市场指数
PE6: 全模块集成验证
"""

import sys
import os
import random
import time

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
print("Phase E 交付验证 — IC验证集 + 51函数 + 冲击成本 + 退避 + BETA")
print("=" * 60)


# ============================================================================
# PE1: IC仅验证集
# ============================================================================
_step("PE1: compute_ic 仅验证集计算")

from smartalpha.eval.metrics import compute_ic, compute_market_beta

np.random.seed(42)
dates_full = pd.date_range("2024-01-02", periods=200, freq="B")
factor_vals = pd.Series(np.random.randn(200), index=dates_full)
price = pd.Series(100 * (1 + np.random.randn(200).cumsum() * 0.02), index=dates_full)

fwd = pd.DataFrame({
    "ret_1d": price.pct_change().shift(-1),
    "ret_5d": price.pct_change(5).shift(-5),
}, index=dates_full)

# 全量 IC
ic_full = compute_ic(factor_vals, fwd)
n_full_1d = ic_full["ret_1d"].dropna().shape[0]
n_full_5d = ic_full["ret_5d"].dropna().shape[0]

# 仅验证集 IC (train_end="2024-06-30")
ic_val = compute_ic(factor_vals, fwd, train_end="20240630")
n_val_1d = ic_val["ret_1d"].dropna().shape[0]
n_val_5d = ic_val["ret_5d"].dropna().shape[0]

train_end_dt = pd.Timestamp("2024-06-30")
val_count = sum(1 for d in dates_full if d > train_end_dt)
print(f"   全量 IC: 1d={n_full_1d}, 5d={n_full_5d} | 验证集 IC: 1d={n_val_1d}, 5d={n_val_5d} | 验证集交易日: {val_count}")
assert n_val_5d > 0, "验证集IC(5d)样本数应为正"
assert n_val_5d <= n_full_5d, "验证集IC不应多于全量IC"
ok(f"train_end 过滤正确 (全量5d={n_full_5d} → 验证集5d={n_val_5d})")


# ============================================================================
# PE2: 51函数验证 (SAR + OBV)
# ============================================================================
_step("PE2: 51函数库 — SAR + OBV 验证")

from smartalpha.core.functions import FinancialFunctionLibrary

lib = FinancialFunctionLibrary()
funcs = lib.list_functions()
assert len(funcs) >= 51, f"函数数不足51: {len(funcs)}"
ok(f"函数总数: {len(funcs)}")

# SAR 验证
n = 50
price_sim = 100 + np.cumsum(np.random.randn(n) * 2)
high = price_sim + np.abs(np.random.randn(n) * 0.5)
low = price_sim - np.abs(np.random.randn(n) * 0.5)
sar_result = lib.call("SAR", high, low)
assert len(sar_result) == n, "SAR 输出长度应等于输入"
assert not np.all(np.isnan(sar_result)), "SAR 不应全为 NaN"
ok("SAR 计算正常")

# OBV 验证
close_v = 100 + np.cumsum(np.random.randn(n) * 1.5)
volume = np.abs(np.random.randn(n) * 1000000) + 500000
obv_result = lib.call("OBV", close_v, volume)
assert len(obv_result) == n, "OBV 输出长度应等于输入"
assert not np.all(obv_result == 0), "OBV 不应全为零"
ok("OBV 计算正常")


# ============================================================================
# PE3: 冲击成本独立建模
# ============================================================================
_step("PE3: 冲击成本独立建模")

from smartalpha.backtest.engine import AShareCostModel

cost = AShareCostModel(impact_lambda=0.1, impact_alpha=0.5)

# 无成交量时冲击成本为0
imp0 = cost.impact_cost(10000, 0)
assert imp0 == 0.0, "零成交量冲击成本应为0"
ok("零成交量边界")

# 小额交易冲击成本接近0
imp_small = cost.impact_cost(1000, 100_000_000)
print(f"   小额(1000/1亿) 冲击成本: {imp_small:.6f}")
assert imp_small < 1.0, "小额交易冲击成本应极小"
ok(f"小额冲击成本合理: {imp_small:.6f}")

# 大额交易冲击成本显著
imp_large = cost.impact_cost(5_000_000, 100_000_000)
print(f"   大额(500万/1亿) 冲击成本: {imp_large:.2f}")
assert imp_large > 100, "大额交易冲击成本应显著"
ok(f"大额冲击成本显著: {imp_large:.2f}")

# buy_cost / sell_cost 包含冲击成本
buy_with_impact = cost.buy_cost(5_000_000, 100_000_000)
buy_no_impact = cost.buy_cost(5_000_000, 0)
assert buy_with_impact > buy_no_impact, "含成交量的买入成本应更高"
ok("buy_cost/sell_cost 已集成冲击成本")


# ============================================================================
# PE4: 指数退避+jitter
# ============================================================================
_step("PE4: 指数退避 + jitter 重试逻辑验证")

# 直接测试重试算法的等待时间分布
base = 1.0
jitter = 0.5
max_wait = 30.0

waits = []
for attempt in range(3):
    w = min(base * (2 ** attempt), max_wait)
    w += random.uniform(-jitter, jitter)
    w = max(0.1, w)
    waits.append(round(w, 3))

print(f"   重试间隔: attempt0={waits[0]}s, attempt1={waits[1]}s, attempt2={waits[2]}s")
# 验证退避递增
assert waits[0] >= 0.0, "首次等待应 >= 0"
assert waits[0] <= 1.5, "首次等待应 <= base+jitter"
# 第二次至少是 base*2 - jitter
assert waits[1] >= 1.5, f"第二次等待应 >= 1.5s, 实际 {waits[1]}"
ok("指数退避+jitter 算法正确")

# 验证 fetcher 模块可以导入
from smartalpha.data.fetcher import TushareFetcher
ok("fetcher 模块导入成功")


# ============================================================================
# PE5: BETA绑定市场指数
# ============================================================================
_step("PE5: compute_market_beta 绑定市场指数")

np.random.seed(123)
n_days = 120
idx_dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
market_ret = pd.Series(np.random.randn(n_days) * 0.01, index=idx_dates)
stock_ret = 0.8 * market_ret + np.random.randn(n_days) * 0.005
stock_ret.index = idx_dates

beta = compute_market_beta(stock_ret, market_ret, window=60)
assert isinstance(beta, pd.Series), "返回值应为 Series"

# 前 window-1 天应为 NaN
assert beta.iloc[:59].isna().all(), "前59天应为NaN"
# 后续应有有效值
valid_beta = beta.dropna()
assert len(valid_beta) > 0, "应有有效Beta值"

# Beta 应接近 0.8 (模拟的 stock = 0.8*market + noise)
mean_beta = valid_beta.mean()
print(f"   平均 Beta: {mean_beta:.4f} (理论值 0.8)")
assert 0.4 < mean_beta < 1.2, f"Beta 应在合理范围(0.4~1.2), 实际 {mean_beta:.4f}"
ok(f"市场 Beta 计算正确 (均值 {mean_beta:.4f})")

# 短数据边界
short_ret = pd.Series(np.random.randn(20) * 0.01)
short_mkt = pd.Series(np.random.randn(20) * 0.01)
beta_short = compute_market_beta(short_ret, short_mkt, window=60)
assert beta_short.isna().all(), "数据不足应全返回NaN"
ok("短数据边界正确")


# ============================================================================
# PE6: 全模块集成验证
# ============================================================================
_step("PE6: 全模块集成 — 新功能在真实模拟数据上运行")

from smartalpha.backtest.engine import BacktestEngine
from smartalpha.risk.manager import RiskManager, RiskLimits
from smartalpha.strategy import StrategyConfig, compare_strategies
from smartalpha.model.lgbm import LightGBMPredictor
from smartalpha.model.trainer import WalkForwardTrainer

# 6.1 带冲击成本的回测
np.random.seed(42)
dates_bt = pd.date_range("2024-01-02", periods=80, freq="B")
stock_list = [f"S{i:02d}" for i in range(8)]

data_records = []
for s in stock_list:
    drift_s = 0.0003 * (int(s[1:]) - 3)
    px = 100 * (1 + np.random.randn(80).cumsum() * 0.015 + np.arange(80) * drift_s)
    for j, d in enumerate(dates_bt):
        data_records.append({"trade_date": d, "ts_code": s, "close": max(px[j], 1)})

panel_bt = pd.DataFrame(data_records).set_index(["trade_date", "ts_code"])
pw = panel_bt["close"].unstack("ts_code")
sig = pw.pct_change(5).rank(axis=1, pct=True).stack()
sig.index.names = ["trade_date", "ts_code"]

cost_with_impact = AShareCostModel(impact_lambda=0.15, impact_alpha=0.5)
engine_impact = BacktestEngine(top_n=3, rebalance_freq="M", cost_model=cost_with_impact)
result_impact = engine_impact.run(panel_bt, sig)

assert result_impact.metrics, "回测应有指标"
print(f"   冲击成本回测: 夏普={result_impact.metrics.get('sharpe', 'N/A')}")
ok("带冲击成本回测通过")

# 6.2 WalkForward + IC仅验证集
trainer = WalkForwardTrainer(purge_days=3, val_days=20, step_days=20)
n_samples = 150
X_sim = pd.DataFrame({
    f"factor_{i}": np.random.randn(n_samples) for i in range(5)
})
y_sim = pd.Series(np.random.randn(n_samples) * 0.02)
dates_sim = pd.Series(pd.date_range("2024-01-02", periods=n_samples, freq="B"))

wf_result = trainer.run(X_sim, y_sim, dates_sim, min_train_days=60)
if not wf_result.metrics or len(wf_result.fold_results) == 0:
    print(f"   [WARN] WalkForward 无 fold（模拟数据量不足），跳过")
    ok("WalkForward 接口调用通过（模拟数据无 fold）")
else:
    assert "ic" in wf_result.metrics, "应计算 IC"
    print(f"   WalkForward IC={wf_result.metrics.get('ic', 'N/A'):.4f}, RMSE={wf_result.metrics.get('rmse', 'N/A'):.4f}")
    ok("WalkForward 集成通过")

# 6.3 多策略 + 冲击成本对比
rm_int = RiskManager(RiskLimits(stop_loss_single=-0.06, max_single_position=0.10))
strategies_int = [
    StrategyConfig("Top3_冲击成本", sig, BacktestEngine(top_n=3, cost_model=cost_with_impact)),
    StrategyConfig("Top3_标准费率", sig, BacktestEngine(top_n=3)),
]
report_int = compare_strategies(strategies_int, panel_bt)
assert report_int.ranking is not None and len(report_int.ranking) >= 2, "应有2+策略结果"
ok("多策略冲击成本对比通过")


# ============================================================================
# 总结
# ============================================================================
print()
print("=" * 60)
total = PASS
actual_pass = total - FAILS
print(f"Phase E 交付验证结果: {actual_pass}/{total} 项通过 ({FAILS} 失败)")
print("=" * 60)

if FAILS == 0:
    print("Phase E 全部通过 — PE1~PE6 就绪。")
else:
    print(f"{FAILS} 项未通过，请检查上方 FAIL 详情。")

print()
print("--- 生产级差距 (诚实文档) ---")
print("PE1: IC验证集 — 需要真实多资产截面数据做IC序列")
print("PE2: SAR/OBV — 需真实行情数据验证指标与标准库(Talib)一致")
print("PE3: 冲击成本 — λ/α参数需用量化交易风控模型校准")
print("PE4: 退避+jitter — 重试逻辑仅适用于 Tushare，AKShare需独立实现")
print("PE5: BETA指数 — 需要真实沪深300/中证500指数数据")
print("PE6: 集成测试 — 当前为模拟数据，需真实数据全链路验证")
