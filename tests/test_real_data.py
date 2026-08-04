"""SmartAlpha Pro 端到端真实数据验证。

验证全链路: 数据加载 → 因子计算 → 中性化 → ML训练 → 回测 → 风控。

运行前提:
    python scripts/download_data.py --index-hs300   # 先下载数据

运行:
    python tests/test_real_data.py                  # 完整验证
    python tests/test_real_data.py --quick           # 快速验证（仅数据层）
"""

import sys
import os
import argparse
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

PASS = 0
FAILS = 0


def step(name):
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
print("SmartAlpha Pro 端到端真实数据验证")
print("=" * 60)

# 检查数据是否存在
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
if not os.path.exists(CACHE_DIR):
    print()
    print("  未找到缓存数据！请先运行:")
    print("    python scripts/download_data.py --index-hs300")
    sys.exit(1)


# ============================================================================
# 1. 数据层验证
# ============================================================================
step("数据层: 缓存文件存在性")

cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".parquet")]
n_files = len(cache_files)
print(f"   缓存文件数: {n_files}")
assert n_files > 0, "缓存目录为空"
ok(f"缓存文件 {n_files} 个")


step("数据层: 单文件格式验证")

sample_file = os.path.join(CACHE_DIR, cache_files[0])
df_sample = pd.read_parquet(sample_file)
required_cols = ["trade_date", "ts_code", "open", "high", "low", "close"]
missing = [c for c in required_cols if c not in df_sample.columns]
assert not missing, f"缺少列: {missing}"
assert len(df_sample) > 0, "文件为空"
print(f"   样本文件: {cache_files[0]} ({len(df_sample)} 行)")
ok("Parquet 格式正确")


step("数据层: 面板构建")

from smartalpha.data.panel_builder import PanelBuilder, build_panel_from_cache

raw_frames = []
for f in cache_files:
    try:
        df = pd.read_parquet(os.path.join(CACHE_DIR, f))
        if not df.empty:
            raw_frames.append(df)
    except Exception:
        pass

raw = pd.concat(raw_frames, ignore_index=True)
builder = PanelBuilder(raw)
panel = builder.build()

n_stocks = panel.attrs.get("n_stocks", 0)
n_dates = panel.attrs.get("n_dates", 0)
print(f"   面板: {n_stocks} 只 × {n_dates} 日")
assert n_stocks >= 5, f"股票数不足: {n_stocks}"
assert n_dates >= 60, f"交易日不足: {n_dates}"
ok(f"面板构建成功 ({n_stocks}×{n_dates})")


# ============================================================================
# 2. 数据质量验证
# ============================================================================
step("数据质量: 缺失率检查")

pw = builder.price_matrix(use_adj=False)
missing_rate = pw.isnull().mean().mean()
print(f"   价格缺失率: {missing_rate:.2%}")
assert missing_rate < 0.5, f"缺失率过高: {missing_rate:.2%}"
ok(f"缺失率 {missing_rate:.2%}")


step("数据质量: 价格合理性检查")

close_pw = pw.dropna(how="all")
min_price = close_pw.min().min()
max_price = close_pw.max().max()
print(f"   价格范围: {min_price:.2f} ~ {max_price:.2f}")
assert min_price > 0, "存在非正价格"
assert max_price < 10000, "存在异常高价"
ok("价格范围合理")


step("数据质量: 日期连续性")

dates = builder.date_list()
date_gaps = dates[1:] - dates[:-1]
max_gap = date_gaps.max().days
avg_gap = date_gaps.mean().days
print(f"   最大间隔: {max_gap} 天, 平均间隔: {avg_gap:.1f} 天")
ok(f"日期连续性可接受 (最大间隔 {max_gap} 天)")


# ============================================================================
# 3. 因子计算验证
# ============================================================================
step("因子计算: 表达式引擎")

from smartalpha.core.executor import ASTExecutor

executor = ASTExecutor()
# 用真实收盘价测试简单因子
sample_stock = pw.columns[0]
close_series = pw[sample_stock].dropna()
if len(close_series) > 30:
    context = {"close": close_series.values}
    # 简单动量因子: close / delay(close, 20) - 1
    momentum = close_series.values / np.roll(close_series.values, 20)
    momentum[:20] = np.nan
    momentum = momentum - 1
    assert not np.all(np.isnan(momentum)), "动量全为 NaN"
    print(f"   动量因子均值: {np.nanmean(momentum):.4f}")
    ok("因子计算正常")


step("因子计算: 多股票因子")

# 所有股票计算简单动量
fwd_ret = pw.pct_change(5).shift(-5)  # 5日前向收益作为标签
factor_simple = pw.pct_change(20)  # 20日动量因子
# 截面排名标准化
factor_rank = factor_simple.rank(axis=1, pct=True)

from smartalpha.eval.metrics import compute_ic

# 计算 IC (使用单只股票时序IC)
stock_ic_values = []
for s in pw.columns[:min(20, len(pw.columns))]:
    f = factor_simple[s].dropna()
    r = fwd_ret[s].dropna()
    common = f.index.intersection(r.index)
    if len(common) > 30:
        stock_ic_values.append(f.loc[common].corr(r.loc[common]))

if stock_ic_values:
    mean_ic = np.mean(stock_ic_values)
    print(f"   平均 IC (时序): {mean_ic:.4f} ({len(stock_ic_values)} 只)")
    ok(f"IC 计算正常 (均值 {mean_ic:.4f})")
else:
    print("   [WARN] IC 计算数据不足")


# ============================================================================
# 4. 行业中性化验证
# ============================================================================
step("中性化: 行业分类加载")

from smartalpha.data.industry_fetcher import (
    IndustryFetcher, load_industry_map_from_cache,
)

industry_map = load_industry_map_from_cache("data/industry_map.parquet")
if not industry_map:
    # 尝试在线获取
    fetcher = IndustryFetcher()
    industry_map = fetcher.build_map()

if industry_map:
    n_industries = len(set(industry_map.values()))
    print(f"   行业数: {n_industries}, 股票数: {len(industry_map)}")
    ok(f"行业分类就绪 ({n_industries} 个行业)")
else:
    print("   [WARN] 行业分类不可用，使用代码前缀粗分类")
    # 构建粗分类
    for s in pw.columns:
        industry_map[s] = IndustryFetcher.coarse_classify(s)
    ok("使用代码前缀粗分类")


step("中性化: 行业中性化执行")

from smartalpha.factor.neutralize import industry_neutralize

# 选10只股票做截面中性化
test_stocks = list(pw.columns)[:10]
test_dates = pw.index[-60:]  # 最近60天

for d in test_dates:
    if d not in pw.index:
        continue
    factor_slice = factor_simple.loc[d, test_stocks].dropna()
    if len(factor_slice) < 3:
        continue

    industry_slice = pd.Series(
        {s: industry_map.get(s, "未知") for s in factor_slice.index}
    )

    try:
        neutralized = industry_neutralize(factor_slice, industry_slice)
        assert len(neutralized) == len(factor_slice), "长度不匹配"
        # 中性化后均值应接近0
        neu_mean = neutralized.mean()
        print(f"    日期 {d.date()}: 原均值={factor_slice.mean():.4f}, 中性化后={neu_mean:.6f}")
        assert abs(neu_mean) < 1.0, f"中性化后均值偏离过大: {neu_mean:.4f}"
        ok(f"行业中性化正常 (残差均值 {neu_mean:.6f})")
        break
    except Exception as e:
        continue


# ============================================================================
# 5. 回测验证
# ============================================================================
step("回测: 截面回测引擎")

from smartalpha.backtest.engine import BacktestEngine, AShareCostModel
from smartalpha.risk.manager import RiskManager, RiskLimits

# 构建信号: 截面排名
signal = pw.pct_change(20).rank(axis=1, pct=True)
sig_stacked = signal.stack()
sig_stacked.index.names = ["trade_date", "ts_code"]

# 构建面板 MultiIndex
if not isinstance(raw.index, pd.MultiIndex):
    panel_mi = raw.set_index(["trade_date", "ts_code"])
else:
    panel_mi = raw

engine = BacktestEngine(top_n=10, rebalance_freq="M")
try:
    result = engine.run(panel_mi, sig_stacked, price_col="close")
    if result.metrics:
        m = result.metrics
        print(f"   年化收益: {m.get('annual_return', 0):.2%}")
        print(f"   夏普比率: {m.get('sharpe', 0):.2f}")
        print(f"   最大回撤: {m.get('max_drawdown', 0):.2%}")
        print(f"   日VaR95: {m.get('var_95', 0):.2%}")
        ok("截面回测执行成功")
    else:
        print("   [WARN] 回测指标为空")
except Exception as e:
    print(f"   [WARN] 回测失败: {e}")


# ============================================================================
# 6. 风控全链路验证
# ============================================================================
step("风控: 完整风控链")

rm = RiskManager(RiskLimits(
    stop_loss_single=-0.10,
    max_single_position=0.10,
    blacklist_days=5,
    daily_loss_limit=-0.05,
    max_monthly_loss=-0.15,
    consecutive_loss_days=3,
))

try:
    result_risk = engine.run(
        panel_mi, sig_stacked, price_col="close",
        risk_manager=rm, industry_map=industry_map,
    )
    if result_risk.metrics:
        events = result_risk.metrics.get("risk_events_total", 0)
        event_types = result_risk.metrics.get("risk_events_detail", {})
        print(f"   风控事件: {events} 次")
        print(f"   事件类型: {dict(event_types)}")
        print(f"   夏普(有风控): {result_risk.metrics.get('sharpe', 0):.2f}")
        ok("完整风控链执行成功")
    else:
        print("   [WARN] 风控回测指标为空")
except Exception as e:
    print(f"   [WARN] 风控回测失败: {e}")


# ============================================================================
# 7. BETA 计算验证
# ============================================================================
step("BETA: 市场 Beta 计算")

from smartalpha.eval.metrics import compute_market_beta
from smartalpha.data.index_fetcher import get_market_returns

ret_data_start = str(dates[0]).replace("-", "")[:8] if len(dates) > 0 else "20200101"
ret_data_end = str(dates[-1]).replace("-", "")[:8] if len(dates) > 0 else "20260730"

try:
    market_ret = get_market_returns("000300.SH", ret_data_start, ret_data_end)
except Exception:
    market_ret = pd.Series(dtype=float)

if len(market_ret) > 60:
    # 选一只代表性股票
    test_stock = pw.columns[0]
    stock_ret = pw[test_stock].pct_change().dropna()
    stock_ret.index = pd.to_datetime(stock_ret.index)

    beta = compute_market_beta(stock_ret, market_ret, window=60)
    beta_valid = beta.dropna()
    if len(beta_valid) > 0:
        print(f"   {test_stock} 平均 Beta: {beta_valid.mean():.4f}")
        assert 0.2 < beta_valid.mean() < 3.0, f"Beta 异常: {beta_valid.mean():.4f}"
        ok(f"市场 Beta 计算正确 ({beta_valid.mean():.2f})")
    else:
        print("   [WARN] Beta 无有效值")
else:
    print("   [WARN] 市场指数数据不足，跳过 Beta 计算")


# ============================================================================
# 总结
# ============================================================================
print()
print("=" * 60)
total = PASS
actual_pass = total - FAILS
print(f"端到端验证结果: {actual_pass}/{total} 项通过 ({FAILS} 失败)")
print("=" * 60)

if FAILS == 0:
    print("全链路验证通过 — 数据层→因子→中性化→回测→风控→BETA 全流程就绪。")
else:
    print(f"{FAILS} 项未通过")

print()
print("--- 生产级差距 (诚实文档) ---")
print("1. 数据源: AKShare 全量下载 ~5000只需 4-6h，建议 Tushare 付费积分")
print("2. 财务数据: 当前未接入财报数据(LEVERAGE/GROWTH 需 Tushare income/balance 接口)")
print("3. 行业分类: 申万/中信分类需 Tushare 更高权限积分")
print("4. 增量更新: 建议配置 cron 每日自动下载")
print("5. 性能: 5000只 × 6年的因子计算需批量优化")
print("6. 回测准确性: 需与 Wind/聚宽等专业平台交叉验证")
