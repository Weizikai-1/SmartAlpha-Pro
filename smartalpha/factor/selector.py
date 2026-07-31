"""因子选择管道 — IC筛选 + 相关性去重 + 截面相关系数筛选。

将海量候选因子缩减为少量有效、低相关的因子集合。

管道:
1. IC筛选:    仅保留 |IC均值| >= min_abs_ic 的因子
2. 相关性去重: 对高相关因子对(>max_corr)，保留IC更高的那个
3. 截面相关筛选: 移除与其他因子截面相关系数均值过高的因子

生产级数据需求 (诚实文档):
- 需要真实的前向收益率数据才能计算有效IC。
- IC计算应在验证集(OOF日期区间)上执行，避免全样本数据窥探。
- 当前 selector 的 IC 计算在传入的全部数据上执行，
  生产环境中需与 WalkForwardTrainer 的 OOF 分割配合使用。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartalpha.eval.metrics import compute_ic
from smartalpha._constants import EPS


def filter_by_ic(
    factors: pd.DataFrame,
    forward_returns: pd.Series,
    min_abs_ic: float = 0.02,
    min_periods: int = 20,
) -> pd.DataFrame:
    """按 IC 绝对值筛选因子。

    对每个因子，计算其与未来收益的相关系数 (全样本 Pearson IC)。
    保留 |IC| >= min_abs_ic 的因子。

    Args:
        factors: date×factor 的因子值宽表，或 factor×stock 表。
                 支持两种格式: 按截面传入(每行=截面)或按时序传入。
        forward_returns: 前向收益率，index 需与 factors 对齐。
        min_abs_ic: IC 绝对值最低阈值 (默认 0.02)。
        min_periods: 最少数据点数 (低于此数视为不可靠，直接丢弃)。

    Returns:
        筛选后的因子 DataFrame，仅保留有效列。

    数据要求 (生产级):
        - forward_returns 必须是样本外(OOF)数据区间的前向收益。
        - 不要在训练集上计算 IC，会导致选择偏差。
    """
    if factors.empty or forward_returns.empty:
        return factors

    if len(factors) < min_periods or len(forward_returns) < min_periods:
        return pd.DataFrame(index=factors.index)

    kept_columns = []
    ic_records = {}

    for col in factors.columns:
        factor_vals = factors[col].dropna()

        # 对齐
        common_idx = factor_vals.index.intersection(forward_returns.dropna().index)
        if len(common_idx) < min_periods:
            continue

        f = factor_vals.loc[common_idx]
        r = forward_returns.loc[common_idx]

        ic = f.corr(r)  # Pearson IC

        if pd.isna(ic):
            continue

        ic_records[col] = round(float(ic), 6)

        if abs(ic) >= min_abs_ic:
            kept_columns.append(col)

    if not kept_columns:
        return pd.DataFrame(index=factors.index)

    # 按 |IC| 降序排列
    kept_columns.sort(key=lambda c: abs(ic_records.get(c, 0)), reverse=True)

    result = factors[kept_columns].copy()
    result.attrs["ic_records"] = ic_records
    return result


def remove_correlated(
    factors: pd.DataFrame,
    max_corr: float = 0.70,
    method: str = "pearson",
) -> pd.DataFrame:
    """移除高相关因子 (保留 IC 更高者)。

    算法:
    1. 计算因子间相关系数矩阵
    2. 对相关系数 > max_corr 的因子对，保留 attributes["ic_records"] 中 IC 绝对值更高的
    3. 若 attributes 中无 ic_records，则保留排在前面的

    Args:
        factors: date×factor 宽表。
        max_corr: 相关系数阈值，超过此值的因子对去重 (默认 0.70)。
        method: 相关方法 ("pearson" 或 "spearman")。

    Returns:
        去重后的因子 DataFrame。
    """
    if factors.empty or len(factors.columns) < 2:
        return factors

    # 先剔除方差为零的列 (相关系数无意义)
    stds = factors.std()
    valid_cols = [c for c in factors.columns if stds.get(c, 0) > EPS]
    if len(valid_cols) < 2:
        return factors[valid_cols]

    corr_matrix = factors[valid_cols].corr(method=method)

    # 获取 IC 记录 (若存在)
    ic_records = factors.attrs.get("ic_records", {})
    if ic_records:
        def _score(col):
            return abs(ic_records.get(col, 0))
    else:
        def _score(col):
            return 1  # 无 IC 信息时保留列顺序靠前的

    # 贪心去重
    columns = list(valid_cols)
    columns.sort(key=_score, reverse=True)
    kept: list[str] = []

    for col in columns:
        conflict = False
        for k in kept:
            corr_val = abs(corr_matrix.loc[col, k])
            if not pd.isna(corr_val) and corr_val > max_corr:
                conflict = True
                break
        if not conflict:
            kept.append(col)

    return factors[kept]


def cross_sectional_corr_filter(
    factors: pd.DataFrame,
    max_mean_corr: float = 0.50,
) -> pd.DataFrame:
    """截面相关系数均值筛选。

    对每个因子，计算它与其他所有因子的相关系数均值。
    移除均值 > max_mean_corr 的因子 (表示该因子信息已被其他因子充分覆盖)。

    Args:
        factors: date×factor 宽表。
        max_mean_corr: 截面相关系数均值上限 (默认 0.50)。

    Returns:
        筛选后的因子 DataFrame。
    """
    if factors.empty or len(factors.columns) < 2:
        return factors

    valid_cols = [c for c in factors.columns if factors[c].std() > EPS]
    if len(valid_cols) < 2:
        return factors[valid_cols]

    corr_matrix = factors[valid_cols].corr()

    kept = []
    for col in valid_cols:
        others = [c for c in valid_cols if c != col]
        if not others:
            kept.append(col)
            continue
        mean_corr = float(corr_matrix.loc[col, others].abs().mean())
        if mean_corr <= max_mean_corr:
            kept.append(col)

    return factors[kept]


def select_factors(
    factors: pd.DataFrame,
    forward_returns: pd.Series | None = None,
    min_abs_ic: float = 0.02,
    max_corr: float = 0.70,
    max_mean_corr: float = 0.50,
    min_periods: int = 20,
) -> pd.DataFrame:
    """一站式因子选择管道: IC筛选 → 相关性去重 → 截面相关筛选。

    Args:
        factors: date×factor 宽表。
        forward_returns: 前向收益率 (用于IC筛选，可选)。
                         为 None 时跳过 IC 筛选。
        min_abs_ic: IC 阈值。
        max_corr: 相关性去重阈值。
        max_mean_corr: 截面相关系数均值阈值。
        min_periods: IC 筛选的最少数据点数。

    Returns:
        筛选后的因子 DataFrame。

    数据要求 (生产级):
        - forward_returns 必须是 OOF (样本外) 区间的数据。
        - 建议与 WalkForwardTrainer 配合，在每期 OOF 区间独立做因子选择。
    """
    result = factors.copy()

    # Step 1: IC 筛选
    if forward_returns is not None and not forward_returns.empty:
        result = filter_by_ic(result, forward_returns, min_abs_ic, min_periods)
        if result.empty:
            return result

    # Step 2: 相关性去重
    result = remove_correlated(result, max_corr)
    if result.empty:
        return result

    # Step 3: 截面相关系数均值筛选
    result = cross_sectional_corr_filter(result, max_mean_corr)

    return result
