"""涨跌停 Mask 过滤 — 剔除无法交易日的因子值。

A股有 ±10% (科创/创业 ±20%) 涨跌停限制。涨停时无法买入，
跌停时无法卖出，这些日的价格不反映真实供需，因子值无意义。

策略:
- 根据日收益率绝对值是否 >= 阈值判断涨跌停
- 将对应日期的因子值置 NaN
- 下游函数 (IC计算、选股) 中 NaN 自动被忽略

生产级数据需求 (诚实文档):
- 需要每只股票的历史日线 OHLCV 数据 (已通过 DataLoader 支持)。
- 阈值: 主板 9.5%, 科创/创业板 19.5%, 北交所 29.5%。
  当前版本使用统一阈值 9.5%，覆盖主板情况。
  精确实现需结合 stock_basic 中的交易所/板块信息动态设置阈值。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartalpha._constants import LIMIT_THRESHOLD


def build_limit_mask(
    price: pd.DataFrame,
    threshold: float = LIMIT_THRESHOLD,
    price_col: str = "close",
) -> pd.DataFrame:
    """构建涨跌停 Mask (True=可交易, False=涨跌停)。

    涨跌停判断: |当日收益率| >= threshold

    Args:
        price: 价格数据。
               MultiIndex(date, stock) 单列，或 date×stock 的宽表。
        threshold: 涨跌停阈值 (默认 0.095 = 9.5%)。
        price_col: 价格列名 (仅 MultiIndex 格式)。

    Returns:
        bool DataFrame，date × stock，True 表示正常交易日。

    数据要求 (生产级):
        - 需要真实日线数据，pct_change 计算基于复权价格
        - 创业板/科创板需单独设置 threshold=0.195
    """
    if price.empty:
        return pd.DataFrame()

    # 统一转为宽表 (date × stock)
    if isinstance(price.index, pd.MultiIndex):
        price_wide = price[price_col].unstack("ts_code")
    else:
        price_wide = price

    ret = price_wide.pct_change().abs()
    mask = ret < threshold
    # 首日无收益率，设为可交易
    mask.iloc[0] = True

    return mask


def apply_mask(
    factor: pd.DataFrame | pd.Series,
    mask: pd.DataFrame,
) -> pd.DataFrame | pd.Series:
    """将 Mask 应用到因子值 (涨跌停日置 NaN)。

    Args:
        factor: 因子值，index 可以是:
                - MultiIndex(date, stock) 的单列 Series
                - date×stock 的宽表 DataFrame
        mask: build_limit_mask 返回的 bool 宽表。

    Returns:
        与 factor 同格式的掩码后数据。
    """
    if mask.empty:
        return factor

    # 统一为宽表
    if isinstance(factor.index, pd.MultiIndex):
        factor_wide = factor.unstack("ts_code")
        is_multiindex = True
    else:
        factor_wide = factor
        is_multiindex = False

    # 对齐 index/columns
    common_dates = factor_wide.index.intersection(mask.index)
    common_stocks = factor_wide.columns.intersection(mask.columns)
    if len(common_dates) == 0 or len(common_stocks) == 0:
        return factor

    f_aligned = factor_wide.loc[common_dates, common_stocks]
    m_aligned = mask.loc[common_dates, common_stocks]

    # NaN 处保留，False 处置 NaN
    result_wide = f_aligned.where(m_aligned)

    if is_multiindex:
        result = result_wide.stack()
        result.index.names = ["trade_date", "ts_code"]
        return result
    else:
        # 补回不对齐的行/列
        result_wide_full = factor_wide.copy()
        result_wide_full.loc[common_dates, common_stocks] = result_wide.loc[common_dates, common_stocks]
        return result_wide_full
