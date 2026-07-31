"""多策略对比模块 — 同时运行多组策略，输出对比报告。"""

from .comparator import StrategyConfig, ComparisonReport, compare_strategies

__all__ = ["StrategyConfig", "ComparisonReport", "compare_strategies"]
