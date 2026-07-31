"""smartalpha.eval — 因子评估与回测模块。"""

from .metrics import evaluate_factor, compute_ic, compute_sharpe, compute_max_drawdown
from .reporter import FactorReport

__all__ = [
    "evaluate_factor",
    "compute_ic",
    "compute_sharpe",
    "compute_max_drawdown",
    "FactorReport",
]
