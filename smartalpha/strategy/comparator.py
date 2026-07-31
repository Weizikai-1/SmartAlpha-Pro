"""多策略对比基准 — 同时运行多组策略，输出对比报告。

解决 L7 要求: 必须有真多策略对比，而非仅与等权基准比较。

使用示例:
    from smartalpha.strategy import Strategy, compare_strategies
    strategies = [
        Strategy("动量Top10", signal_momentum, BacktestEngine(top_n=10)),
        Strategy("低波Top10", signal_lowvol, BacktestEngine(top_n=10)),
        Strategy("等权基准", None, None, is_benchmark=True),
    ]
    report = compare_strategies(strategies, panel)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from smartalpha.backtest.engine import BacktestEngine, BacktestResult


@dataclass
class StrategyConfig:
    """单个策略的配置。

    Attributes:
        name: 策略名称。
        signal: 截面信号 Series，为 None 时使用等权选股。
        engine: 回测引擎实例，为 None 时使用默认 Top20/月度调仓。
        risk_manager: 可选的风控管理器。
        industry_map: 可选行业映射。
        is_benchmark: 是否为基准策略（等权持有所有股票）。
    """

    name: str
    signal: Optional[pd.Series] = None
    engine: Optional[BacktestEngine] = None
    risk_manager: Optional[object] = None
    industry_map: Optional[dict] = None
    is_benchmark: bool = False

    def get_engine(self) -> BacktestEngine:
        if self.engine is not None:
            return self.engine
        return BacktestEngine(top_n=20, rebalance_freq="M")


@dataclass
class ComparisonReport:
    """多策略对比报告。"""

    results: dict[str, BacktestResult] = field(default_factory=dict)
    ranking: pd.DataFrame | None = None

    def summary(self) -> str:
        if self.ranking is not None:
            lines = ["=" * 72, "多策略对比报告", "=" * 72]
            lines.append(self.ranking.to_string(index=False))
            lines.append(f"\n最佳策略: {self.ranking.iloc[0]['策略']}")
            return "\n".join(lines)
        return "无有效回测结果"


def compare_strategies(
    strategies: list[StrategyConfig],
    panel: pd.DataFrame,
    price_col: str = "close",
) -> ComparisonReport:
    """执行多策略对比。

    对每个策略用同一面板数据运行回测，生成对比排名表。

    Args:
        strategies: 策略配置列表。
        panel: 面板数据 (MultiIndex: date × stock)。
        price_col: 价格列名。

    Returns:
        ComparisonReport，含各策略绩效和排名。

    数据要求 (生产级):
        - 所有策略必须在同一天的面板数据上运行，确保可比性。
        - 等权基准的信号设为 None，引擎内部自动等权配置。
    """
    results = {}
    metrics_list = []

    for cfg in strategies:
        engine = cfg.get_engine()

        if cfg.is_benchmark:
            price_w = panel[price_col].unstack("ts_code")
            mi = pd.MultiIndex.from_product(
                [price_w.index, price_w.columns],
                names=["trade_date", "ts_code"],
            )
            signal = pd.Series(1.0, index=mi)
        else:
            signal = cfg.signal

        if signal is None or signal.empty:
            results[cfg.name] = BacktestResult()
            continue

        result = engine.run(
            panel, signal, price_col=price_col,
            risk_manager=cfg.risk_manager,
            industry_map=cfg.industry_map,
        )
        results[cfg.name] = result

        if result.metrics and "sharpe" in result.metrics:
            m = result.metrics
            metrics_list.append({
                "策略": cfg.name,
                "年化收益": f"{m.get('annual_return', 0):.2%}",
                "夏普": f"{m.get('sharpe', 0):.2f}",
                "最大回撤": f"{m.get('max_drawdown', 0):.2%}",
                "Calmar": f"{m.get('calmar', 0):.2f}",
                "日VaR95": f"{m.get('var_95', 0):.2%}",
                "日胜率": f"{m.get('daily_win_rate', 0):.1%}",
                "风控事件": str(m.get("risk_events_total", 0)),
            })

    if metrics_list:
        ranking = pd.DataFrame(metrics_list)
        ranking["_sharpe_sort"] = [
            float(str(r["夏普"])) for r in metrics_list
        ]
        ranking = ranking.sort_values("_sharpe_sort", ascending=False)
        ranking = ranking.drop(columns=["_sharpe_sort"])
    else:
        ranking = pd.DataFrame()

    return ComparisonReport(results=results, ranking=ranking)
