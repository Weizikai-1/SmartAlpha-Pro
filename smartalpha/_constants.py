"""向后兼容重导出 — 所有常量已迁移至 smartalpha/config.py。

新代码请直接使用:
    from smartalpha.config import TRADING_DAYS_PER_YEAR, EPS, ...
"""

from smartalpha.config import (  # noqa: F401
    TRADING_DAYS_PER_YEAR,
    EPS,
    LIMIT_THRESHOLD,
    MAX_MISSING_RATIO,
    MIN_DATA_POINTS,
)
