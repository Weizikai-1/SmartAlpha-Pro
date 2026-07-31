"""smartalpha.risk — 风控模块。

提供：
- VaR/CVaR (历史模拟/参数法/蒙特卡洛)
- 止损/止盈机制 (固定+移动)
- 仓位限制 + 行业集中度监控
- 压力测试 (历史极端场景)
"""

from .var import VaRCalculator
from .manager import RiskManager
from .stress import StressTester

__all__ = ["VaRCalculator", "RiskManager", "StressTester"]
