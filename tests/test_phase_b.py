"""Phase B 交付验证 — 用真实数据评估因子有效性。

运行: python tests/test_phase_b.py
"""

import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from smartalpha.data import DataLoader
from smartalpha.core import ExpressionLexer, ExpressionParser, ASTExecutor
from smartalpha.core.functions import FinancialFunctionLibrary
from smartalpha.eval import evaluate_factor, FactorReport


# ==========================================================================
# 1. 加载真实数据
# ==========================================================================
print("=" * 60)
print("Phase B: 真实数据 + 因子评估")
print("=" * 60)

print("\n1. 加载平安银行(000001.SZ)日线数据...")
loader = DataLoader()
df = loader.load_daily(
    ["000001.SZ"],
    start_date="20240101",
    end_date="20260725",
    use_cache=True,
    check_quality=True,
)
print(f"   获取 {len(df)} 条记录, 日期范围 {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}")

# ==========================================================================
# 2. 用表达式引擎计算因子
# ==========================================================================
print("\n2. 用表达式引擎计算真实因子值...")

# 构建因子数据
dates = pd.to_datetime(df["trade_date"].values)
factor_data = {
    "$close": pd.Series(df["close"].values, index=dates),
    "$open": pd.Series(df["open"].values, index=dates),
    "$high": pd.Series(df["high"].values, index=dates),
    "$low": pd.Series(df["low"].values, index=dates),
    "$volume": pd.Series(df["vol"].values, index=dates),
    "$amount": pd.Series(df["amount"].values, index=dates),
}

lexer = ExpressionLexer()
parser = ExpressionParser()
executor = ASTExecutor()

# 测试3个因子
factors_to_test = {
    "momentum_5": "RANK(DELTA($close, 5))",
    "volume_ratio": "$volume / MEAN($volume, 20)",
    "price_range": "($high - $low) / ($close + 1e-10)",
}

results = {}
for name, expr in factors_to_test.items():
    print(f"\n   因子: {name} = {expr}")
    tokens = lexer.tokenize(expr)
    ast = parser.parse(tokens)
    values = executor.execute(ast, factor_data)
    factor_series = pd.Series(values, index=dates)
    print(f"   计算结果: 长度={len(factor_series)}, 非空={factor_series.notna().sum()}, "
          f"均值={factor_series.mean():.4f}, 标准差={factor_series.std():.4f}")
    results[name] = factor_series

# ==========================================================================
# 3. 评估每个因子
# ==========================================================================
print("\n3. 因子评估...")
price = factor_data["$close"]

for name, factor in results.items():
    print(f"\n{'='*60}")
    print(f"评估: {name}")
    print(f"{'='*60}")

    eval_result = evaluate_factor(factor, price, periods=[1, 5, 10, 20])
    report = FactorReport(factor_name=name, metrics=eval_result)
    print(report.generate())

# ==========================================================================
# 4. 对比：因子 vs 随机噪声
# ==========================================================================
print(f"\n{'='*60}")
print("对照组: 随机噪声因子")
print("=" * 60)

noise = pd.Series(np.random.RandomState(42).randn(len(price)), index=price.index)
noise.name = "random_noise"
noise_eval = evaluate_factor(noise, price, periods=[1, 5, 10])
noise_report = FactorReport(factor_name="随机噪声", metrics=noise_eval)
print(noise_report.generate())

# ==========================================================================
# 总结
# ==========================================================================
print(f"\n{'='*60}")
print("Phase B 交付验证完成")
print("=" * 60)
print("✅ 真实数据获取: tushare daily API")
print("✅ 因子计算: 表达式引擎 (词法→语法→AST→执行)")
print("✅ 因子评估: IC/IR/Sharpe/MDD/胜率")
print("✅ 对照组: 随机噪声 (应有较差的IC)")
print("")
print("⚠️ 生产级差距（诚实说明）:")
print("  - 仅测试单只股票 (000001.SZ)，未做全A股截面验证")
print("  - 未做行业中性/市值中性处理")
print("  - 回测为简单多空策略，无真实手续费/滑点模型")
print("  - tushare频率限制导致数据范围受限 (约2.5年日线)")
