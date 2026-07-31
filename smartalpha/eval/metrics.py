"""因子评估指标模块。

基于真实行情数据计算因子的有效性指标。
不做回测，不虚构收益——只算因子与未来收益的统计关系。

计算指标:
- IC (Information Coefficient):    因子值与未来收益的相关系数
- RankIC:                          因子排名与未来收益排名的相关系数  
- Sharpe:                          年化夏普比率
- MaxDrawdown:                     最大回撤
- AnnualReturn:                    年化收益率
- HitRate:                         IC为正的交易日占比
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smartalpha._constants import EPS, TRADING_DAYS_PER_YEAR


def compute_forward_returns(
    price: pd.Series,
    periods: list[int] | None = None,
    purge_days: int = 0,
) -> pd.DataFrame:
    """计算前向收益率（支持 purge 间隔避免标签边界泄漏）。

    标签逻辑:
        ret_p_d at index t = (price[t + purge + p] / price[t + purge]) - 1

    若 purge_days=0，退化为标准前向收益:
        ret_p_d at index t = (price[t + p] / price[t]) - 1

    Args:
        price: 价格序列（按日期升序）。
        periods: 持有期列表，默认 [1, 5, 10]。
        purge_days: 清空间隔天数，标签窗口在 t+purge 后开始。

    Returns:
        DataFrame，每列为一个持有期的前向收益率。
    """
    if periods is None:
        periods = [1, 5, 10]

    result = pd.DataFrame(index=price.index)
    for p in periods:
        if purge_days > 0:
            # 先用 purge 偏移价格序列，再计算前向收益
            # ret at t = pct_change of price[t+purge] to price[t+purge+p]
            shifted = price.shift(-purge_days)
            result[f"ret_{p}d"] = shifted.pct_change(periods=p).shift(-p)
        else:
            result[f"ret_{p}d"] = price.pct_change(periods=p).shift(-p)
    return result


def compute_ic(
    factor: pd.Series,
    forward_returns: pd.DataFrame,
    method: str = "pearson",
    train_end: str | None = None,
) -> pd.DataFrame:
    """计算因子IC序列（仅验证集）。

    对每个交易日截面，计算因子值与未来收益的相关系数。
    若指定 train_end，仅对 train_end 之后的日期（验证集）计算 IC，
    避免训练集污染评估指标。

    Args:
        factor: 因子值（index=日期, values=因子值）。
        forward_returns: 前向收益率（每列一个持有期）。
        method: 相关系数方法 ('pearson' 或 'spearman')。
        train_end: 训练集截止日期 (YYYYMMDD)，None 表示全量。

    Returns:
        IC序列DataFrame，每列一个持有期。
    """
    ic_data = {}

    for col in forward_returns.columns:
        ic_series = []
        ic_dates = []

        common_dates = factor.dropna().index.intersection(
            forward_returns[col].dropna().index
        )
        # 仅验证集: 过滤 train_end 之后的日期
        if train_end is not None:
            train_end_dt = pd.Timestamp(train_end)
            common_dates = common_dates[common_dates > train_end_dt]

        if len(common_dates) < 20:
            continue

        f = factor.loc[common_dates]
        r = forward_returns[col].loc[common_dates]

        corr = f.corr(r, method=method)
        ic_series.append(corr)
        ic_dates.append(common_dates[-1])

        ic_data[col] = pd.Series(ic_series, index=ic_dates)

    return pd.DataFrame(ic_data)


def compute_ic_stats(ic_df: pd.DataFrame) -> dict[str, dict]:
    """计算IC统计量。

    对于单只股票(时序)，IC为全样本相关系数(单个值)。
    对于多只股票(截面)，IC为每日截面相关系数序列。

    Args:
        ic_df: IC序列DataFrame（输出自 compute_ic）。

    Returns:
        {持有期: {ic_mean, ic_std, ic_ir, hit_rate, positive_days, total_days}}
    """
    stats = {}
    for col in ic_df.columns:
        ic = ic_df[col].dropna()
        if len(ic) < 1:
            continue

        ic_mean = float(np.mean(ic))
        ic_std = float(np.std(ic, ddof=1)) if len(ic) > 1 else 0.0
        ic_ir = round(ic_mean / ic_std, 4) if ic_std > EPS else 0.0
        stats[col] = {
            "ic_mean": round(ic_mean, 6),
            "ic_std": round(ic_std, 6),
            "ic_ir": ic_ir,
            "hit_rate": round(float(np.mean(ic > 0)), 4),
            "positive_days": int(np.sum(ic > 0)),
            "total_days": len(ic),
            "is_full_sample": len(ic) == 1,  # 单只股票全样本IC标记
        }
    return stats


def compute_sharpe(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """计算年化夏普比率。

    Args:
        daily_returns: 日收益率序列。
        rf: 无风险利率（默认0）。

    Returns:
        年化夏普比率。
    """
    returns = daily_returns.dropna()
    if len(returns) < 20:
        return 0.0

    excess = returns - rf / TRADING_DAYS_PER_YEAR
    if excess.std() < EPS:
        return 0.0

    # 年化：日夏普 * sqrt(252)
    return float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """计算最大回撤。

    Args:
        equity_curve: 净值曲线（如 (1+returns).cumprod()）。

    Returns:
        最大回撤（负值，-0.2 表示回撤20%）。
    """
    equity = equity_curve.dropna()
    if len(equity) < 2:
        return 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    return float(drawdown.min())


def compute_annual_return(daily_returns: pd.Series) -> float:
    """计算年化收益率。

    Args:
        daily_returns: 日收益率序列。

    Returns:
        年化收益率。
    """
    returns = daily_returns.dropna()
    if len(returns) < 1:
        return 0.0

    cum = (1 + returns).prod()
    periods = len(returns)
    return float(cum ** (TRADING_DAYS_PER_YEAR / periods) - 1)


def compute_turnover(daily_positions: pd.Series) -> float:
    """计算日均换手率。

    Args:
        daily_positions: 每日持仓权重变化（绝对值之和/2）。

    Returns:
        日均换手率。
    """
    changes = daily_positions.diff().abs().dropna()
    if len(changes) < 1:
        return 0.0
    return float(changes.mean() / 2)


def compute_market_beta(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """计算个股对市场指数的滚动 Beta。

    Beta = Cov(stock, market) / Var(market)
    仅使用市场指数（如 000300.SH 沪深300）作为基准，
    确保 Beta 因子具有经济含义。

    Args:
        stock_returns: 个股日收益率序列。
        market_returns: 市场指数日收益率序列（必须对齐 index）。
        window: 滚动窗口，默认 60 个交易日。

    Returns:
        Beta 值序列（index 与输入对齐，前 window-1 行为 NaN）。
    """
    common_idx = stock_returns.dropna().index.intersection(
        market_returns.dropna().index
    )
    if len(common_idx) < window:
        return pd.Series(np.nan, index=stock_returns.index)

    sr = stock_returns.loc[common_idx]
    mr = market_returns.loc[common_idx]

    beta = pd.Series(np.nan, index=stock_returns.index)

    for i in range(window - 1, len(common_idx)):
        idx = common_idx[i]
        s_win = sr.iloc[i - window + 1 : i + 1]
        m_win = mr.iloc[i - window + 1 : i + 1]
        cov = np.cov(s_win, m_win, ddof=1)
        var_m = np.var(m_win, ddof=1)
        if var_m > EPS:
            beta.loc[idx] = cov[0, 1] / var_m

    return beta


def evaluate_factor(
    factor: pd.Series,
    price: pd.Series,
    periods: list[int] | None = None,
) -> dict:
    """一站式因子评估。

    Args:
        factor: 因子值序列（同index同price）。
        price: 价格序列。
        periods: 评估的持有期列表。

    Returns:
        包含全部评估指标的字典。
    """
    if periods is None:
        periods = [1, 5, 10, 20]

    # 对齐索引
    common_idx = factor.dropna().index.intersection(price.dropna().index)
    factor = factor.loc[common_idx]
    price = price.loc[common_idx]

    # 前向收益
    fwd = compute_forward_returns(price, periods)

    # IC
    ic_df = compute_ic(factor, fwd, method="pearson")
    rank_ic_df = compute_ic(factor.rank(pct=True), fwd, method="spearman")

    # 基于因子值的简单多空收益（用expanding window避免未来信息泄漏）
    exp_mean = factor.expanding(min_periods=60).mean()
    exp_std = factor.expanding(min_periods=60).std()
    exp_std = exp_std.replace(0, 1)  # 避免除零
    factor_z = ((factor - exp_mean) / exp_std).clip(-2, 2)
    daily_pos = factor_z.fillna(0)
    price_ret = price.pct_change()
    daily_ret = daily_pos.shift(1) * price_ret  # 今仓×明收益

    return {
        "factor_name": (factor.name if hasattr(factor, "name") else ""),
        "data_period": f"{factor.index[0]} ~ {factor.index[-1]}",
        "data_points": len(factor),
        "ic": compute_ic_stats(ic_df),
        "rank_ic": compute_ic_stats(rank_ic_df),
        "sharpe": round(compute_sharpe(daily_ret), 4),
        "max_drawdown": round(compute_max_drawdown((1 + daily_ret).cumprod()), 4),
        "annual_return": round(compute_annual_return(daily_ret), 4),
        "turnover": round(compute_turnover(daily_pos), 4),
    }


def label_purge_check(
    forward_returns: pd.DataFrame,
    train_end: str,
    periods: list[int] | None = None,
) -> dict:
    """前向标签泄漏检查。

    检测 forward returns 是否跨越 train/test 边界。
    例如: ret_5d 在 train 最后一天 t 的标签使用了 price[t+5]，
    如果 t+5 属于 test 区间，则造成数据泄漏。

    Args:
        forward_returns: compute_forward_returns 的输出，每列一个持有期。
        train_end: 训练集截止日期 (YYYYMMDD)。
        periods: 各列对应的持有期列表，默认从列名解析 (如 ret_5d → 5)。

    Returns:
        检查报告字典:
        - clean: 是否有泄漏 (bool)
        - leaked_dates: 各持有期的泄漏日期列表
        - safe_dates: 各持有期的安全日期列表
        - recommended_purge: 各持有期建议的 purge 天数
    """
    if periods is None:
        periods = []
        for col in forward_returns.columns:
            # 从列名 "ret_Xd" 解析持有期
            try:
                periods.append(int(col.replace("ret_", "").replace("d", "")))
            except ValueError:
                periods.append(1)

    train_end_dt = pd.Timestamp(train_end)
    report = {"clean": True, "leaked_dates": {}, "safe_dates": {}, "recommended_purge": {}}

    for col, period in zip(forward_returns.columns, periods):
        # 前向标签 at date t uses price[t+period]
        # 安全范围: t + period <= train_end → t <= train_end - period
        safe_boundary = train_end_dt - pd.Timedelta(days=period)

        all_dates = forward_returns[col].dropna().index
        leaked = all_dates[all_dates > safe_boundary]
        safe = all_dates[all_dates <= safe_boundary]

        report["leaked_dates"][col] = list(leaked)
        report["safe_dates"][col] = list(safe)
        # 建议 purge 天数 = 最大持有期 (保守方案)
        report["recommended_purge"][col] = period

        if len(leaked) > 0:
            report["clean"] = False

    return report
