"""项目级常量定义。

避免散落在各模块中的魔法数字，统一管理与复用。
"""

# 交易日常数
TRADING_DAYS_PER_YEAR = 252

# 数值精度 epsilon (避免除零)
EPS = 1e-10

# 涨跌停阈值 (A股±10%)
LIMIT_THRESHOLD = 0.095

# 数据质量
MAX_MISSING_RATIO = 0.1
MIN_DATA_POINTS = 20
