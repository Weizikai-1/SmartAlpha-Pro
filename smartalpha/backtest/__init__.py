"""smartalpha.backtest — 截面回测引擎。

支持多股票截面选股、A股真实费率模型、组合绩效评估。
"""

from .engine import AShareCostModel, BacktestEngine, CrossSectionBacktest

__all__ = ["AShareCostModel", "BacktestEngine", "CrossSectionBacktest"]
