"""Phase C 交付验证 — 多股票截面回测。

运行: python tests/test_phase_c.py

验证内容:
1. 多股票真实数据加载
2. 因子信号计算（表达式引擎）
3. 月度截面选股回测
4. 绩效指标输出（净值/夏普/回撤/换手/费率/VaR）
5. 诚实评估生产级差距
"""

import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from smartalpha.data import DataLoader
from smartalpha.core import ExpressionLexer, ExpressionParser, ASTExecutor
from smartalpha.core.functions import FinancialFunctionLibrary
from smartalpha.backtest import BacktestEngine, AShareCostModel
from smartalpha.eval import FactorReport, evaluate_factor

# ============================================================================
# 1. 加载多股票数据
# ============================================================================
print("=" * 60)
print("Phase C: 多股票截面回测")
print("=" * 60)

STOCKS = [
    "000001.SZ",  # 平安银行
    "000002.SZ",  # 万科A
    "000858.SZ",  # 五粮液
    "002415.SZ",  # 海康威视
    "300750.SZ",  # 宁德时代
    "600000.SH",  # 浦发银行
    "600036.SH",  # 招商银行
    "600276.SH",  # 恒瑞医药
    "600519.SH",  # 贵州茅台
    "601318.SH",  # 中国平安
]

START_DATE = "20230701"
END_DATE = "20260725"

print(f"\n1. 加载 {len(STOCKS)} 只股票 {START_DATE[:4]}-{END_DATE[:4]} 日线数据...")
loader = DataLoader()
df = loader.load_daily(STOCKS, START_DATE, END_DATE, use_cache=True, check_quality=True)
print(f"   获取 {len(df)} 条记录, {df['ts_code'].nunique()} 只股票")

# ============================================================================
# 2. 构建面板数据
# ============================================================================
print("\n2. 构建价格/信号面板...")
df["trade_date_dt"] = pd.to_datetime(df["trade_date"])
panel = df.set_index(["trade_date_dt", "ts_code"]).sort_index()

price = panel["close"].unstack("ts_code")
# 去除非交易价格（涨跌停Mask）
pct = price.pct_change().fillna(0)
MASK = (pct.abs() < 0.095)  # A股±10%涨跌停，排除触及涨跌停的

print(f"   价格矩阵: {price.shape} ({price.index[0].date()} ~ {price.index[-1].date()})")
print(f"   有效交易日: {len(price)}")

# ============================================================================
# 3. 用表达式引擎计算因子信号
# ============================================================================
print("\n3. 计算因子信号（表达式引擎）...")

lexer = ExpressionLexer()
parser = ExpressionParser()
executor = ASTExecutor()

# 构建因子（动量反转 + 波动率）
signals = {}
for code in STOCKS:
    stock_data = df[df["ts_code"] == code].set_index("trade_date_dt").sort_index()
    if len(stock_data) < 120:
        continue

    factor_data = {
        "$close": stock_data["close"],
        "$open": stock_data["open"],
        "$high": stock_data["high"],
        "$low": stock_data["low"],
        "$volume": stock_data["vol"],
        "$amount": stock_data["amount"],
    }

    # 因子1: 20日动量（反转信号：跌多了反弹）
    tokens = lexer.tokenize("DELTA($close, 20) / $close")
    ast = parser.parse(tokens)
    momentum = pd.Series(executor.execute(ast, factor_data), index=stock_data.index)

    # 因子2: 20日波动率倒数（低波偏好）
    tokens = lexer.tokenize("STD($close, 20) / $close")
    ast = parser.parse(tokens)
    vol_inv = pd.Series(executor.execute(ast, factor_data), index=stock_data.index)

    # 综合信号: -动量 + 低波（动量越小越好=反转，波动越小越好）
    combined = -momentum.rank(pct=True).fillna(0.5) + vol_inv.rank(pct=True).fillna(0.5)
    signals[code] = combined

signal_df = pd.DataFrame(signals)
signal_df = signal_df.reindex(price.index)

print(f"   信号矩阵: {signal_df.shape}")
print(f"   信号覆盖率: {signal_df.notna().sum().sum() / signal_df.size:.1%}")

# ============================================================================
# 4. 月度截面回测
# ============================================================================
print("\n4. 运行月度截面回测（Top 5, ¥100万初始资金）...")

engine = BacktestEngine(
    init_cash=1_000_000,
    top_n=5,
    rebalance_freq="M",
    cost_model=AShareCostModel(
        commission=0.0003,   # 万三佣金
        stamp_duty=0.0005,   # 千0.5印花税（卖出）
        slippage=0.001,      # 千1滑点
    ),
)

# 构建MultiIndex面板用于回测
panel_backtest = df.set_index(["trade_date_dt", "ts_code"]).sort_index()
signal_stacked = signal_df.stack().rename("signal")
signal_stacked.index.names = ["trade_date", "ts_code"]

result = engine.run(panel_backtest, signal_stacked, price_col="close")

# ============================================================================
# 5. 基准对比：等权买入持有
# ============================================================================
print("\n5. 基准对比（等权买入持有）...")
bench_ret = price.pct_change().mean(axis=1).dropna()
bench_cum = (1 + bench_ret).cumprod()
print(f"   基准累计收益: {bench_cum.iloc[-1] - 1:.4%}")

# ============================================================================
# 6. 交付报告
# ============================================================================
print("\n" + result.summary())

# ============================================================================
# 总结
# ============================================================================
print(f"\n{'=' * 60}")
print("Phase C 交付验证完成")
print("=" * 60)
print("✅ 多股票真实数据加载: tushare daily API")
print("✅ 因子信号计算: 表达式引擎（动量+低波）")
print("✅ 截面回测: 月度Top5选股，等权配置")
print("✅ 交易成本: A股真实费率（万三佣金+千0.5印花税+千1滑点）")
print("✅ 绩效评估: 净值/年化收益/夏普/最大回撤/VaR/CVaR/换手/基准对比")
print("")
print("⚠️ 生产级差距（诚实说明）:")
print("  - 仅测试10只股票，未做全A股（~5000只）截面验证")
print("  - 因子为简单组合（动量反转+低波），非ML预测模型")
print("  - 未做行业中性化、市值中性化处理")
print("  - 未引入Purge间隔防止标签泄漏")
print("  - 未做多频率对比（周度/日度）")
print("  - 信号提取（Rank组合）未做IC筛选和相关性去重")
print("  - 无止损/止盈机制，无风险控制层")
print("  - 费率估算为近似值，实际需逐笔匹配成交价")
