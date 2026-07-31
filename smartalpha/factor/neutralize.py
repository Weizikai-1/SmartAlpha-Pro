"""因子中性化 — 行业+市值回归残差法。

消除因子中的行业偏向和规模偏向，使因子信号更"纯粹"。

方法:
- 行业中性化: 对每个交易日截面，因子值 ~ 行业哑变量 回归，取残差。
- 市值中性化: 因子值 ~ log(市值) 回归，取残差。
- 组合中性化: 因子值 ~ 行业哑变量 + log(市值) 多元回归，取残差。

生产级数据需求 (诚实文档):
- 行业分类: 申万一级行业或证监会行业分类，需通过Tushare/AKShare获取。
- 市值数据: A股总市值/流通市值，需通过数据源获取。
- 当前版本用 numpy.linalg.lstsq 实现，适合中等规模数据(<5000只股票)。
  超大规模数据建议用 statsmodels 或 sklearn 的稀疏矩阵优化。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def industry_neutralize(
    factor: pd.Series,
    industry: pd.Series,
) -> pd.Series:
    """行业中性化 — 截面回归取残差。"""
    if factor.empty or industry.empty:
        return factor.copy()

    common_idx = factor.index.intersection(industry.index)
    if len(common_idx) == 0:
        return factor.copy()

    factor = factor.loc[common_idx]
    industry = industry.loc[common_idx].astype(str)

    if isinstance(factor.index, pd.MultiIndex) and "trade_date" in factor.index.names:
        return _cross_sectional_neutralize(factor, industry)
    else:
        return _single_section_neutralize(factor, industry)


def market_cap_neutralize(
    factor: pd.Series,
    market_cap: pd.Series,
) -> pd.Series:
    """市值中性化 — 对 log(市值) 回归取残差。

    对每个交易日截面:
        factor_i = α + β × log(market_cap_i) + ε_i

    Args:
        factor: MultiIndex(date, stock) 因子值。
        market_cap: 与 factor 同 index 的市值序列。

    Returns:
        中性化后的因子值。
    """
    if factor.empty or market_cap.empty:
        return factor.copy()

    common_idx = factor.index.intersection(market_cap.index)
    if len(common_idx) == 0:
        return factor.copy()

    factor = factor.loc[common_idx]
    mc = market_cap.loc[common_idx]

    # 对数市值 (剔除零/负值)
    log_mc = np.log(mc.replace(0, np.nan))
    log_mc = log_mc.replace([np.inf, -np.inf], np.nan)

    if isinstance(factor.index, pd.MultiIndex) and "trade_date" in factor.index.names:
        return _cross_sectional_neutralize(factor, log_mc)
    else:
        return _single_section_neutralize(factor, log_mc)


def neutralize(
    factor: pd.Series,
    industry: pd.Series | None = None,
    market_cap: pd.Series | None = None,
) -> pd.Series:
    """组合中性化 — 行业 + 市值 多元回归取残差。

    factor_i = α + Σ(β_j × I_j) + γ × log(market_cap_i) + ε_i

    Args:
        factor: 因子值。
        industry: 行业标签 (可选，None 则跳过行业中性化)。
        market_cap: 市值 (可选，None 则跳过市值中性化)。

    Returns:
        中性化后的因子值。
    """
    if industry is None and market_cap is None:
        return factor

    # 构建所有回归变量
    regressors = []

    # 对齐: 找到 industry 和 market_cap 中与 factor 共同的 index
    if industry is not None and market_cap is not None:
        # 三路对齐
        idx_i = set(industry.dropna().index)
        idx_m = set(market_cap.dropna().index)
        idx_f = set(factor.dropna().index)
        common = sorted(idx_f & idx_i & idx_m)
        if not common:
            return factor
        factor = factor.loc[common]
        cols = [industry.loc[common].astype(str)]
        mc = market_cap.loc[common]
        log_mc = np.log(mc.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        cols.append(log_mc)
    elif industry is not None:
        common = sorted(set(factor.dropna().index) & set(industry.dropna().index))
        if not common:
            return factor
        factor = factor.loc[common]
        cols = [industry.loc[common]]
    else:
        common = sorted(set(factor.dropna().index) & set(market_cap.dropna().index))
        if not common:
            return factor
        factor = factor.loc[common]
        mc = market_cap.loc[common]
        log_mc = np.log(mc.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        cols = [log_mc]

    if isinstance(factor.index, pd.MultiIndex) and "trade_date" in factor.index.names:
        return _cross_sectional_neutralize(factor, *cols)
    else:
        return _single_section_neutralize(factor, *cols)


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------

def _single_section_neutralize(
    y: pd.Series, *regressors: pd.Series
) -> pd.Series:
    """单截面最小二乘残差。"""
    # 构建设计矩阵
    X_parts = []
    for r in regressors:
        # 判断是否为数值类型
        try:
            is_numeric = np.issubdtype(r.dtype, np.number)
        except TypeError:
            is_numeric = False

        if not is_numeric:
            # 分类变量 → 哑变量 (确保 str 类型)
            dummies = pd.get_dummies(r.astype(str), drop_first=True).astype(float)
            if dummies.shape[1] > 0:
                X_parts.append(dummies.values)
        else:
            X_parts.append(r.values.reshape(-1, 1))

    X = np.column_stack(X_parts) if len(X_parts) > 1 else X_parts[0]
    y_vals = y.values.astype(float)

    # —— NaN/Inf 强化清理 ——
    # 1. 剔除 y 中的 NaN/Inf
    y_finite = np.isfinite(y_vals)
    # 2. 剔除 X 中的 NaN/Inf
    x_finite = np.all(np.isfinite(X), axis=1)
    # 3. 剔除 X 中过于极端的值 (1e15 以上视为数据异常)
    x_bounded = np.all(np.abs(X) < 1e15, axis=1)
    valid = y_finite & x_finite & x_bounded

    if valid.sum() < 3:
        return pd.Series(np.nan, index=y.index, dtype=float)

    X_v = X[valid]
    y_v = y_vals[valid]

    # 检查 X_v 是否满秩 (列间无 NaN，且行数 ≥ 列数)
    if X_v.shape[0] < X_v.shape[1] + 1:
        return pd.Series(np.nan, index=y.index, dtype=float)

    try:
        coeffs, residuals, rank, singular = np.linalg.lstsq(X_v, y_v, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return pd.Series(np.nan, index=y.index, dtype=float)

    y_pred = X_v @ coeffs
    residual = y_v - y_pred

    # 拼回原 Index
    result = pd.Series(np.nan, index=y.index, dtype=float)
    result.iloc[np.where(valid)[0]] = residual
    return result


def _cross_sectional_neutralize(
    y: pd.Series, *regressors: pd.Series
) -> pd.Series:
    """逐日期截面中性化 (MultiIndex 版本)。"""
    result = y.copy()
    dates = y.index.get_level_values("trade_date").unique()

    for date in dates:
        y_date = y.xs(date, level="trade_date", drop_level=False)
        # 取对应截面
        reg_date = []
        has_nan = False
        for r in regressors:
            try:
                r_slice = r.loc[y_date.index]
            except KeyError:
                has_nan = True
                break
            reg_date.append(r_slice)

        if has_nan or y_date.dropna().empty:
            continue

        try:
            resid = _single_section_neutralize(y_date, *reg_date)
        except Exception:
            import logging
            logging.getLogger(__name__).warning(f"截面中性化失败: {date}")
            continue

        if not resid.dropna().empty:
            result.loc[resid.index] = resid

    return result
