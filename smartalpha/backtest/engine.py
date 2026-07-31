"""截面回测引擎 — 多股票组合回测，含A股真实费率模型。

设计原则:
- 数据先行: 只接受已加载的真实数据，不自取数据。
- 评估先行: 每次回测自动输出完整的绩效指标。
- 严格时序: 用shift(1)保证信号日T的决策对应T+1收益，无未来信息泄漏。
- 诚实费率: A股真实佣金+印花税+滑点，不虚构成本。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from smartalpha._constants import EPS, TRADING_DAYS_PER_YEAR


# ============================================================================
# A股交易费率模型
# ============================================================================

@dataclass
class AShareCostModel:
    """A股真实交易费率，含冲击成本独立建模。

    Attributes:
        commission: 佣金费率（双边，默认万三）。
        stamp_duty: 印花税（卖出单边，默认千0.5）。
        slippage: 滑点（单边，默认千1）。
        min_commission: 单笔最低佣金（元），默认5元。
        impact_lambda: 冲击成本系数 λ，默认 0.1。
        impact_alpha: 冲击成本指数 α，默认 0.5（平方根模型）。
    """

    commission: float = 0.0003
    stamp_duty: float = 0.0005
    slippage: float = 0.001
    min_commission: float = 5.0
    impact_lambda: float = 0.1
    impact_alpha: float = 0.5

    def impact_cost(
        self, trade_amount: float, avg_daily_volume: float
    ) -> float:
        """冲击成本（独立建模）。

        基于平方根模型估算大额交易对市场的冲击:
            impact = λ × (trade_amount / avg_daily_volume)^α

        Args:
            trade_amount: 交易金额（元）。
            avg_daily_volume: 该股票日均成交额（元）。

        Returns:
            冲击成本（元）。
        """
        if avg_daily_volume <= 0 or trade_amount <= 0:
            return 0.0
        participation_rate = trade_amount / avg_daily_volume
        return float(self.impact_lambda * (participation_rate ** self.impact_alpha) * trade_amount)

    def buy_cost(
        self, trade_amount: float, avg_daily_volume: float = 0.0
    ) -> float:
        """买入成本 = 佣金 + 滑点 + 冲击成本。"""
        fee = max(trade_amount * self.commission, self.min_commission)
        fee += trade_amount * self.slippage
        fee += self.impact_cost(trade_amount, avg_daily_volume)
        return fee

    def sell_cost(
        self, trade_amount: float, avg_daily_volume: float = 0.0
    ) -> float:
        """卖出成本 = 佣金 + 印花税 + 滑点 + 冲击成本。"""
        fee = max(trade_amount * self.commission, self.min_commission)
        fee += trade_amount * self.stamp_duty
        fee += trade_amount * self.slippage
        fee += self.impact_cost(trade_amount, avg_daily_volume)
        return fee

    def round_trip_cost_ratio(self) -> float:
        """一次完整买入+卖出的费率比例（不计最小费和冲击成本）。"""
        return 2 * self.commission + self.stamp_duty + 2 * self.slippage


# ============================================================================
# 回测结果
# ============================================================================

@dataclass
class BacktestResult:
    """回测结果，包含净值曲线和绩效指标。

    Attributes:
        nav: 日度净值序列（index=date）。
        daily_returns: 日度收益率序列（扣除费用后）。
        turnover: 逐日换手率序列。
        positions: 逐日持仓DataFrame（date × stock_code）。
        metrics: 绩效指标字典。
    """

    nav: Optional[pd.Series] = None
    daily_returns: Optional[pd.Series] = None
    turnover: Optional[pd.Series] = None
    positions: Optional[pd.DataFrame] = None
    metrics: dict = field(default_factory=dict)
    _warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """生成回测结果摘要。"""
        m = self.metrics
        if not m:
            return "空回测结果（无交易）"
        lines = [
            "=" * 60,
            "回测绩效报告",
            "=" * 60,
            f"回测区间: {m.get('start_date', 'N/A')} → {m.get('end_date', 'N/A')}",
            f"交易日数: {m.get('trading_days', 'N/A')}",
            f"初始资金: ¥{m.get('init_cash', 0):,.0f}",
            f"最终净值: ¥{m.get('final_nav', 0):,.2f}",
            "",
            "--- 收益指标 ---",
            f"累计收益: {m.get('cum_return', 0):.4%}",
            f"年化收益: {m.get('annual_return', 0):.4%}",
            f"年化夏普: {m.get('sharpe', 0):.4f}",
            f"最大回撤: {m.get('max_drawdown', 0):.4%}",
            f"Calmar比率: {m.get('calmar', 0):.4f}",
            "",
            "--- 风险指标 ---",
            f"年化波动: {m.get('annual_volatility', 0):.4%}",
            f"日VaR 95%: {m.get('var_95', 0):.4%}",
            f"日CVaR 95%: {m.get('cvar_95', 0):.4%}",
            "",
            "--- 交易指标 ---",
            f"日均换手: {m.get('avg_turnover', 0):.4%}",
            f"总交易费: ¥{m.get('total_cost', 0):,.2f}",
            f"交易费占比: {m.get('cost_ratio', 0):.4%}",
            f"胜率(日): {m.get('daily_win_rate', 0):.4%}",
            f"盈亏比: {m.get('profit_loss_ratio', 'N/A')}",
            "",
            f"--- 基准对比 (买入持有) ---",
            f"基准年化收益: {m.get('benchmark_return', 0):.4%}",
            f"超额收益(Alpha): {m.get('excess_return', 0):.4%}",
            f"信息比率(IR): {m.get('information_ratio', 0):.4f}",
        ]
        if self._warnings:
            lines.append("")
            lines.append("--- 警告 ---")
            for w in self._warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)


# ============================================================================
# 回测引擎
# ============================================================================

class BacktestEngine:
    """截面回测引擎。

    在每个调仓日:
    1. 根据截面信号对所有股票排序
    2. 选前 top_n 只股票等权配置
    3. 计算调仓成本并记录净值

    使用示例:
        engine = BacktestEngine(init_cash=1_000_000, top_n=20, rebalance_freq="M")
        result = engine.run(panel_data, signal_series)
        print(result.summary())
    """

    def __init__(
        self,
        init_cash: float = 1_000_000,
        top_n: int = 20,
        rebalance_freq: str = "M",
        cost_model: Optional[AShareCostModel] = None,
    ) -> None:
        """初始化回测引擎。

        Args:
            init_cash: 初始资金（元）。
            top_n: 每次调仓选入的股票数量。
            rebalance_freq: 调仓频率，'D'=日度, 'W'=周度, 'M'=月度。
            cost_model: 交易费率模型，默认A股标准费率。
        """
        self.init_cash = init_cash
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
        self.cost = cost_model or AShareCostModel()

    # ------------------------------------------------------------------
    # 主执行入口
    # ------------------------------------------------------------------

    def run(
        self,
        panel: pd.DataFrame,
        signal: pd.Series,
        price_col: str = "close",
        benchmark_col: Optional[str] = None,
        risk_manager: Optional[object] = None,
        industry_map: Optional[dict] = None,
    ) -> BacktestResult:
        """执行截面回测（可选集成风控管理器）。

        Args:
            panel: 面板数据，MultiIndex(date, stock_code) 或
                   date为index、stock_code为column的DataFrame。
            signal: 截面信号序列，index与panel对应。
            price_col: panel中用作价格的列名（仅MultiIndex格式）。
            benchmark_col: 基准价格列名，None则用所有股票等权组合。
            risk_manager: RiskManager 实例，为 None 则不启用风控。
            industry_map: 股票→行业 映射，启用行业风控时必传。

        Returns:
            BacktestResult 包含净值曲线和绩效指标。

        严格时序保证:
            信号(T) → 持仓(T) → 收益(T+1)。
            T日收盘后用T日已知信号决定T+1日持仓，
            T+1日收盘价计算收益 = T+1持仓 × T+1日收益。

        风控集成 (新增):
            每个交易日，在计算当日收益后调用 risk_manager.check()。
            触发止损/止盈/仓位限制/行业集中度检查，
            调整后的权重覆盖当日原有权重。
        """
        # 预处理数据
        price_matrix = self._build_price_matrix(panel, price_col)
        signal_matrix = self._build_signal_matrix(panel, signal)
        dates = price_matrix.index.sort_values()
        stocks = price_matrix.columns

        if len(dates) < 2:
            import logging
            logging.getLogger(__name__).warning("回测数据不足（<2个交易日）")
            return BacktestResult()

        # 每日收益率矩阵 (T-1 → T)
        ret_matrix = price_matrix.pct_change()

        # 构建调仓日期集合（用period下标，高效查询）
        rb_dates_set = set(self._get_rebalance_dates(dates))

        # 初始化
        nav = pd.Series(np.nan, index=dates, dtype=float)
        nav.iloc[0] = 1.0
        daily_ret = pd.Series(np.nan, index=dates, dtype=float)
        turnover_series = pd.Series(np.nan, index=dates, dtype=float)
        current_weights = pd.Series(0.0, index=stocks)

        # 首日选股
        rm_blacklist = risk_manager.blacklist_stocks if risk_manager else set()
        if dates[0] in rb_dates_set:
            current_weights = self._select_stocks(signal_matrix.loc[dates[0]], stocks, blacklist=rm_blacklist)
        else:
            current_weights = pd.Series(1.0 / len(stocks), index=stocks)

        # 逐日回测
        risk_events: list = []
        for i in range(1, len(dates)):
            date = dates[i]
            prev_date = dates[i - 1]
            prev_nav = nav.loc[prev_date]

            # 当日股票收益
            stock_rets = ret_matrix.loc[date].fillna(0)
            port_ret = float((current_weights * stock_rets).sum())

            if date in rb_dates_set:
                # 调仓日: 选股 → 计算成本 → 更新NAV → 风控检查
                rm_bl = risk_manager.blacklist_stocks if risk_manager else set()
                target_weights = self._select_stocks(signal_matrix.loc[date], stocks, blacklist=rm_bl)
                cost_ratio = self._calc_turnover_cost(current_weights, target_weights)
                turnover_series.loc[date] = float(
                    abs(current_weights - target_weights).sum() / 2
                )
                nav.loc[date] = prev_nav * (1 + port_ret) * (1 - cost_ratio)

                # 风控检查 (调仓后)
                if risk_manager is not None:
                    daily_pnl = (current_weights * stock_rets).to_dict()
                    target_weights, events = risk_manager.check(
                        str(date.date()), target_weights,
                        nav_history=nav.dropna(),
                        daily_pnl=daily_pnl,
                        industry_map=industry_map,
                    )
                    risk_events.extend(events)
                    # 风控可能改变权重，重新计算换手成本
                    cost_ratio2 = self._calc_turnover_cost(current_weights, target_weights)
                    if cost_ratio2 > cost_ratio:
                        nav.loc[date] = prev_nav * (1 + port_ret) * (1 - cost_ratio2)

                current_weights = target_weights
            else:
                # 非调仓日: 更新NAV → 风控检查 (组合止损/移动止盈)
                nav.loc[date] = prev_nav * (1 + port_ret)

                if risk_manager is not None:
                    daily_pnl = (current_weights * stock_rets).to_dict()
                    adjusted, events = risk_manager.check(
                        str(date.date()), current_weights,
                        nav_history=nav.dropna(),
                        daily_pnl=daily_pnl,
                        industry_map=industry_map,
                    )
                    risk_events.extend(events)
                    if not adjusted.equals(current_weights):
                        current_weights = adjusted

            daily_ret.loc[date] = port_ret

        # 清理NaN
        nav = nav.ffill()
        daily_ret = daily_ret.fillna(0)
        turnover_series = turnover_series.fillna(0)

        # 只使用首个调仓日之后的收益计算绩效
        first_rb = min(rb_dates_set) if rb_dates_set else dates[0]
        ret_for_metrics = daily_ret.loc[daily_ret.index >= first_rb]

        # 计算绩效
        metrics = self._compute_metrics(nav, ret_for_metrics, turnover_series, dates)
        # 附加风控统计 (始终输出，即使为0)
        event_types = {}
        for e in risk_events:
            event_types[e.event_type] = event_types.get(e.event_type, 0) + 1
        metrics["risk_events_total"] = len(risk_events)
        metrics["risk_events_detail"] = event_types
        result = BacktestResult(
            nav=nav,
            daily_returns=daily_ret,
            turnover=turnover_series,
            metrics=metrics,
        )
        if risk_events:
            result._warnings = [f"风控事件 {len(risk_events)} 次: {event_types}"]

        # 基准对比
        self._add_benchmark(result, price_matrix, benchmark_col, daily_ret)

        return result

    # ------------------------------------------------------------------
    # 数据预处理
    # ------------------------------------------------------------------

    def _build_price_matrix(self, panel: pd.DataFrame, col: str) -> pd.DataFrame:
        """构建价格矩阵 (dates × stocks)。"""
        if isinstance(panel.index, pd.MultiIndex):
            return panel[col].unstack("ts_code")
        return panel

    def _build_signal_matrix(
        self, panel: pd.DataFrame, signal: pd.Series
    ) -> pd.DataFrame:
        """构建信号矩阵 (dates × stocks)，与价格矩阵对齐。"""
        if signal.empty or not isinstance(signal.index, pd.MultiIndex):
            return pd.DataFrame()
        if isinstance(panel.index, pd.MultiIndex):
            sig_df = signal.unstack("ts_code") if isinstance(signal, pd.Series) else signal
            price_mat = self._build_price_matrix(panel, "close")
            return sig_df.reindex_like(price_mat) if not price_mat.empty else sig_df
        return signal

    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> list:
        """获取调仓日期列表。"""
        if self.rebalance_freq == "D":
            return list(dates)
        series = pd.Series(0, index=dates)
        if self.rebalance_freq == "W":
            grouper = dates.isocalendar().week
        elif self.rebalance_freq == "M":
            grouper = dates.to_period("M")
        else:
            grouper = dates.to_period("M")

        # 每组最后一个交易日
        last_per_group = series.groupby(grouper).apply(lambda x: x.index[-1])
        # 插入第一个交易日
        result = [dates[0]] + sorted(set(last_per_group))
        return result

    # ------------------------------------------------------------------
    # 选股逻辑
    # ------------------------------------------------------------------

    def _select_stocks(
        self, date_signal: pd.Series, stocks: pd.Index,
        blacklist: set | None = None,
    ) -> pd.Series:
        """根据截面信号选股，返回等权权重。

        Args:
            date_signal: 当日所有股票的信号值。
            stocks: 可选的股票列表。
            blacklist: 黑名单股票集合，被列入的股票不可选中。

        Returns:
            持仓权重 Series（已归一化）。
        """
        # 过滤无效信号和黑名单
        valid = date_signal.dropna()
        if blacklist:
            valid = valid[~valid.index.isin(blacklist)]
        if len(valid) == 0:
            return pd.Series(0.0, index=stocks)

        # 按信号降序，选前 top_n
        if len(valid) > self.top_n:
            selected = valid.nlargest(self.top_n).index
        else:
            selected = valid.index

        # 等权
        weights = pd.Series(0.0, index=stocks)
        weights.loc[selected] = 1.0 / len(selected)
        return weights

    # ------------------------------------------------------------------
    # 交易成本
    # ------------------------------------------------------------------

    def _calc_turnover_cost(
        self, old_w: pd.Series, new_w: pd.Series
    ) -> float:
        """计算一次调仓的交易成本比例。

        Args:
            old_w: 旧权重。
            new_w: 新权重。

        Returns:
            总交易成本占组合净值的比例。
        """
        # 单边换手率 = 权重变化的一半
        turnover = float(abs(old_w - new_w).sum() / 2)
        # 往返费率 = 2×佣金 + 印花税(卖) + 2×滑点
        cost_rate = self.cost.round_trip_cost_ratio() * turnover
        return min(cost_rate, 0.05)  # 上限5%，防止极端值

    # ------------------------------------------------------------------
    # 绩效指标
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        nav: pd.Series,
        daily_ret: pd.Series,
        turnover: pd.Series,
        dates: pd.DatetimeIndex,
    ) -> dict:
        """计算全套绩效指标。"""
        nav_clean = nav.dropna()
        ret_clean = daily_ret.dropna()
        n_days = len(ret_clean)

        if n_days < 5:
            return {"error": "数据不足（<5个交易日）"}

        # 收益
        cum_return = float(nav_clean.iloc[-1] / nav_clean.iloc[0] - 1)
        n_years = n_days  / TRADING_DAYS_PER_YEAR
        annual_ret = float((1 + cum_return) ** (1 / n_years) - 1) if n_years > 0 else 0

        # 波动率
        ann_vol = float(ret_clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        # 夏普
        sharpe = float(ret_clean.mean() / ret_clean.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if ret_clean.std() > EPS else 0

        # 最大回撤（净值法）
        peak = nav_clean.expanding().max()
        drawdown = (nav_clean - peak) / peak
        max_dd = float(drawdown.min())

        # Calmar
        calmar = annual_ret / abs(max_dd) if abs(max_dd) > EPS else 0

        # VaR / CVaR
        var_95 = float(ret_clean.quantile(0.05))
        cvar_95 = float(ret_clean[ret_clean <= var_95].mean()) if (ret_clean <= var_95).sum() > 0 else var_95

        # 交易
        avg_turnover = float(turnover.dropna().mean() or 0)
        total_cost = self._estimate_total_cost(nav_clean, turnover)

        # 胜率 / 盈亏比
        win_days = (ret_clean > 0).sum()
        win_rate = float(win_days / n_days)
        avg_win = ret_clean[ret_clean > 0].mean() if win_days > 0 else 0
        avg_loss = abs(ret_clean[ret_clean < 0].mean()) if (n_days - win_days) > 0 else 1
        pl_ratio = f"{avg_win / avg_loss:.2f}" if avg_loss > EPS else "N/A"

        return {
            "start_date": str(dates[0]),
            "end_date": str(dates[-1]),
            "trading_days": n_days,
            "init_cash": self.init_cash,
            "final_nav": float(nav_clean.iloc[-1]),
            "cum_return": cum_return,
            "annual_return": annual_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "annual_volatility": ann_vol,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "avg_turnover": avg_turnover,
            "total_cost": total_cost,
            "cost_ratio": total_cost / self.init_cash,
            "daily_win_rate": win_rate,
            "profit_loss_ratio": pl_ratio,
            "benchmark_return": 0.0,
            "excess_return": 0.0,
            "information_ratio": 0.0,
        }

    def _estimate_total_cost(
        self, nav: pd.Series, turnover: pd.Series
    ) -> float:
        """估算总交易费用（元）。"""
        total = 0.0
        cost_rate = self.cost.round_trip_cost_ratio()
        turn_vals = turnover.dropna()
        for d, t in turn_vals.items():
            if t <= 0:
                continue
            idx = nav.index.get_loc(d)
            if idx > 0:
                nav_t = nav.iloc[idx - 1]
            else:
                nav_t = 1.0
            # 交易金额 = 换手率 × 前日净值
            trade_amount = nav_t * t * self.init_cash
            total += trade_amount * cost_rate
        return total

    # ------------------------------------------------------------------
    # 基准对比
    # ------------------------------------------------------------------

    def _add_benchmark(
        self,
        result: BacktestResult,
        price_matrix: pd.DataFrame,
        benchmark_col: Optional[str],
        daily_ret: pd.Series,
    ) -> None:
        """计算并添加基准对比指标。"""
        if benchmark_col and benchmark_col in price_matrix.columns:
            bench_price = price_matrix[benchmark_col]
        else:
            # 等权组合作为基准
            bench_price = price_matrix.mean(axis=1)

        bench_ret = bench_price.pct_change().dropna()
        common_idx = bench_ret.index.intersection(daily_ret.dropna().index)
        if len(common_idx) < 10:
            return

        b = bench_ret.loc[common_idx]
        p = daily_ret.loc[common_idx]
        n_years = len(b)  / TRADING_DAYS_PER_YEAR

        bench_annual = float((1 + b).prod() ** (1 / n_years) - 1) if n_years > 0 else 0
        result.metrics["benchmark_return"] = bench_annual
        result.metrics["excess_return"] = result.metrics.get("annual_return", 0) - bench_annual

        excess = p - b
        ir = float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS_PER_YEAR)) if excess.std() > EPS else 0
        result.metrics["information_ratio"] = ir


# ============================================================================
# 便捷接口：因子→回测一站式
# ============================================================================

class CrossSectionBacktest:
    """一站式因子→回测接口。

    封装数据加载、因子计算、截面回测全流程。

    使用示例:
        cbt = CrossSectionBacktest(top_n=20, rebalance_freq="M")
        result = cbt.run_from_signal(
            ts_codes=["000001.SZ", "000002.SZ", ...],
            start_date="20240101",
            end_date="20260725",
            expression="$close / DELAY($close, 20) - 1",
        )
        print(result.summary())
    """

    def __init__(
        self,
        top_n: int = 20,
        rebalance_freq: str = "M",
        init_cash: float = 1_000_000,
        cost_model: Optional[AShareCostModel] = None,
    ) -> None:
        self.engine = BacktestEngine(
            init_cash=init_cash,
            top_n=top_n,
            rebalance_freq=rebalance_freq,
            cost_model=cost_model,
        )

    def run_from_signal(
        self,
        panel: pd.DataFrame,
        signal: pd.Series,
        price_col: str = "close",
    ) -> BacktestResult:
        """用外部信号运行回测。

        Args:
            panel: 面板数据（MultiIndex格式，含date×ts_code）。
            signal: 截面信号序列。
            price_col: 面板中价格列名。

        Returns:
            BacktestResult。
        """
        return self.engine.run(panel, signal, price_col)
