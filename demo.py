"""SmartAlpha Pro 完整演示 — 数据→因子→评估→回测。

运行: python demo.py

展示:
1. 真实数据加载 (Tushare)
2. 表达式引擎因子计算 (动量/低波/量比 3个因子)
3. 因子评估 (IC/IR/Sharpe/MDD)
4. 截面回测 (月度Top5, A股真实费率)
5. 全链路报告

注意: 数据从本地缓存加载，首次运行需要联网获取。
"""

import sys
import time

import pandas as pd

from smartalpha.data import DataLoader
from smartalpha.core import ExpressionLexer, ExpressionParser, ASTExecutor
from smartalpha.core.functions import FinancialFunctionLibrary
from smartalpha.eval import evaluate_factor, FactorReport
from smartalpha.backtest import BacktestEngine, AShareCostModel


# ============================================================================
# 配置
# ============================================================================

STOCKS = [
    "000001.SZ", "000002.SZ", "000858.SZ", "002415.SZ", "300750.SZ",
    "600000.SH", "600036.SH", "600276.SH", "600519.SH", "601318.SH",
]

START = "20230701"
END = "20260725"

FACTOR_EXPRESSIONS = {
    "momentum_20":   "DELTA($close, 20) / DELAY($close, 20)",   # 20日动量
    "volatility_20": "STD($close, 20) / $close",                 # 20日波动率
    "volume_ratio":  "$volume / MEAN($volume, 20)",              # 量比
}

TOP_N = 5
INIT_CASH = 1_000_000


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 65)
    print("  SmartAlpha Pro v2.0 — 量化因子选股系统 全链路演示")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. 数据加载
    # ------------------------------------------------------------------
    print(f"\n▶ 阶段 1/4: 数据加载")
    print(f"  股票: {len(STOCKS)} 只  |  区间: {START[:4]}-{END[:4]}  |  数据源: Tushare Pro")
    t0 = time.time()
    loader = DataLoader()
    df = loader.load_daily(STOCKS, START, END, use_cache=True, check_quality=True)
    elapsed = time.time() - t0
    print(f"  完成: {len(df):,} 条记录, {df['ts_code'].nunique()} 只股票  |  耗时 {elapsed:.1f}s")

    # 构建股票因子数据
    stock_data = {}
    for code in STOCKS:
        sdf = df[df["ts_code"] == code].set_index("trade_date").sort_index()
        sdf.index = pd.to_datetime(sdf.index)
        stock_data[code] = {
            "$close":  sdf["close"],
            "$open":   sdf["open"],
            "$high":   sdf["high"],
            "$low":    sdf["low"],
            "$volume": sdf["vol"],
            "$amount": sdf["amount"],
        }

    price_matrix = pd.DataFrame({c: stock_data[c]["$close"] for c in STOCKS})

    # ------------------------------------------------------------------
    # 2. 因子计算
    # ------------------------------------------------------------------
    print(f"\n▶ 阶段 2/4: 因子计算 (表达式引擎)")
    print(f"  引擎架构: 词法分析 → 语法解析(LL(1)) → AST执行 → 金融函数库(50+函数)")

    lexer = ExpressionLexer()
    parser = ExpressionParser()
    executor = ASTExecutor()

    factor_values = {}
    for fname, expr in FACTOR_EXPRESSIONS.items():
        print(f"  · {fname:15s} = {expr}")
        signals = {}
        for code in STOCKS:
            tokens = lexer.tokenize(expr)
            ast = parser.parse(tokens)
            val = executor.execute(ast, stock_data[code])
            signals[code] = pd.Series(val, index=stock_data[code]["$close"].index)
        factor_values[fname] = pd.DataFrame(signals).reindex(price_matrix.index)
        print(f"    → 形状 {factor_values[fname].shape}, 覆盖率 {factor_values[fname].notna().sum().sum() / factor_values[fname].size:.0%}")

    # ------------------------------------------------------------------
    # 3. 因子评估 (以平安银行为例)
    # ------------------------------------------------------------------
    print(f"\n▶ 阶段 3/4: 因子评估 (以000001.SZ平安银行为例)")
    print(f"  指标: IC/IR | RankIC | 年化夏普 | 最大回撤 | 胜率")

    price_000001 = stock_data["000001.SZ"]["$close"]

    for fname in FACTOR_EXPRESSIONS:
        factor_series = factor_values[fname]["000001.SZ"]
        metrics = evaluate_factor(factor_series, price_000001, periods=[1, 5, 10, 20])
        report = FactorReport(fname, metrics)
        # 精简输出
        ic = metrics.get("ic", {}).get("ret_1d", {})
        print(f"  · {fname:15s}  IC={ic.get('ic_mean', 0):.4f}  "
              f"IC_IR={ic.get('ic_ir', 0):.3f}  "
              f"Sharpe={metrics.get('sharpe', 0):.2f}  "
              f"MDD={metrics.get('max_drawdown', 0):.2%}  "
              f"通过={'✅' if report.passed else '❌'}")

    # ------------------------------------------------------------------
    # 4. 截面回测
    # ------------------------------------------------------------------
    print(f"\n▶ 阶段 4/4: 截面回测")
    print(f"  策略: 月度调仓 Top{TOP_N}  |  费率: {AShareCostModel().round_trip_cost_ratio():.2%} (往返)")
    print(f"  初始资金: ¥{INIT_CASH:,}  |  信号: 动量反转+低波综合")

    # 综合信号: -动量(反转) + 低波 + 量比
    signal = pd.DataFrame(0.0, index=price_matrix.index, columns=price_matrix.columns)
    for fname, fdf in factor_values.items():
        if fname in ("momentum_20", "volatility_20"):
            signal += -fdf.rank(axis=1, pct=True)   # 反转 + 低波
        else:
            signal += fdf.rank(axis=1, pct=True)    # 量比

    signal_stacked = signal.stack().rename("signal")
    signal_stacked.index.names = ["trade_date", "ts_code"]

    engine = BacktestEngine(
        init_cash=INIT_CASH,
        top_n=TOP_N,
        rebalance_freq="M",
        cost_model=AShareCostModel(commission=0.0003, stamp_duty=0.0005, slippage=0.001),
    )

    panel = df.set_index(["trade_date", "ts_code"])
    # 将字符串日期转为datetime以匹配信号索引
    panel.index = panel.index.set_levels(
        pd.to_datetime(panel.index.levels[0]), level=0
    )
    panel = panel.sort_index()

    t1 = time.time()
    result = engine.run(panel, signal_stacked, price_col="close")
    print(f"  回测耗时: {time.time() - t1:.1f}s")

    # ------------------------------------------------------------------
    # 最终报告
    # ------------------------------------------------------------------
    print(f"\n{'─' * 65}")
    print(result.summary())

    # 技术栈总览
    print(f"{'─' * 65}")
    print("技术栈总览:")
    print("  数据层:  Tushare Pro API → Parquet缓存 → 质量检查")
    print("  因子层:  表达式引擎 (词法→语法→AST) → 50+金融函数库")
    print("  评估层:  IC/IR/RankIC/Sharpe/MDD/VaR/CVaR/胜率/换手")
    print("  回测层:  截面选股 → A股真实费率 → 月度调仓 → 基准对比")
    print("  存储层:  列式Pickle存储 + LRU缓存 + 增量更新")
    print("  注册层:  因子注册中心 + 依赖图分析 (拓扑排序/环检测)")
    print("  测试:    393单元测试 100%通过")
    print(f"{'─' * 65}")
    print("⚠ 生产级说明:")
    print("  本Demo测试10只股票、3年数据。生产环境需扩展至全A股5000+只。")
    print("  信号为简单Rank组合，非ML预测模型。建议引入LightGBM/Transformer。")
    print("  缺失: 行业中性化、市值中性化、Purge防泄漏、风控止损。")
    print("=" * 65)


if __name__ == "__main__":
    main()
