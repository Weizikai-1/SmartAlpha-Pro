"""因子工程模块 — 中性化、Mask过滤、因子选择。

提供因子从原始值到可交易信号的完整加工管道:
- neutralize:  行业+市值回归残差法中性化
- mask:        涨跌停 Mask 过滤
- selector:    IC筛选 + 相关性去重 + 截面相关系数筛选
"""

from .neutralize import neutralize, industry_neutralize, market_cap_neutralize
from .mask import build_limit_mask, apply_mask
from .selector import (
    filter_by_ic,
    remove_correlated,
    cross_sectional_corr_filter,
    select_factors,
)

__all__ = [
    "neutralize",
    "industry_neutralize",
    "market_cap_neutralize",
    "build_limit_mask",
    "apply_mask",
    "filter_by_ic",
    "remove_correlated",
    "cross_sectional_corr_filter",
    "select_factors",
]
