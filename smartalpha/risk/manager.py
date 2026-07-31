"""风控管理器 — 止损/止盈、仓位限制、行业集中度、黑名单、因子暴露监控。

集成到回测流程中的风控规则:
1. 个股止损: 单只股票亏损超过阈值 → 强制平仓该股 + 加入黑名单
2. 组合止损: 日度组合亏损超过阈值 → 清仓
3. 移动止盈: 从峰值回撤超过阈值 → 减仓 + 加入黑名单
4. 仓位限制: 单只股票 ≤ max_single_position, 单行业 ≤ max_sector
5. 行业集中度: 前N行业权重不超过上限
6. 黑名单: 触发止损后 N 日内禁止再次买入
7. 因子暴露: 监控组合在各因子的暴露度，超阈值报警
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from smartalpha._constants import EPS


@dataclass
class RiskLimits:
    """风控参数配置。"""

    # 止损
    stop_loss_single: float = -0.08     # 个股止损 (-8%)
    stop_loss_portfolio: float = -0.05   # 组合日止损 (-5%)
    stop_loss_sector: float = -0.10      # 行业止损 (-10%)

    # 移动止盈 (从峰值回撤)
    trailing_stop: float = -0.10         # 从最高点回撤10%触发

    # 仓位限制
    max_single_position: float = 0.10    # 单只股票 ≤10%
    max_sector_position: float = 0.30    # 单行业 ≤30%
    max_positions: int = 50              # 最大持仓数

    # 行业集中度
    max_top3_sector: float = 0.60        # 前3行业 ≤60%
    hhi_threshold: float = 0.15          # HHI指数阈值

    # 黑名单 (新增)
    blacklist_days: int = 5              # 触发止损后禁止买入天数

    # 因子暴露 (新增)
    max_factor_exposure: float = 2.0     # 单因子暴露 z-score 上限
    max_total_exposure: float = 5.0      # 总暴露上限

    # 日亏损限额 (新增)
    daily_loss_limit: float = -0.03       # 单日亏损限额 (-3%)
    consecutive_loss_days: int = 3        # 连续亏损天数阈值
    max_monthly_loss: float = -0.10      # 月度亏损限额 (-10%)


@dataclass
class RiskEvent:
    """风控事件记录。"""

    date: str
    event_type: str     # stop_loss / position_limit / sector_limit / trailing_stop
    detail: str
    action: str         # liquidate / reduce / skip


class RiskManager:
    """风控管理器。

    在每个交易日检查组合是否触发风控规则，返回调整后的权重。

    使用示例:
        rm = RiskManager(RiskLimits())
        adjusted_weights, events = rm.check(
            date="2024-01-15",
            weights=current_weights,
            nav_history=nav_series,
            single_pnl={"000001.SZ": -0.12},
            industry_map={"000001.SZ": "银行"},
        )
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self._events: list[RiskEvent] = []
        # 移动止盈跟踪: 股票 → 历史最高权重
        self._peak_weights: dict[str, float] = {}
        # 黑名单: 股票 → 剩余禁止天数
        self._blacklist: dict[str, int] = {}
        # 计数器: 用于日期推进
        self._day_counter: int = 0
        # 日亏损跟踪 (新增): 记录每日组合盈亏
        self._daily_pnl: list[float] = []
        self._consecutive_losses: int = 0
        self._monthly_pnl: float = 0.0
        self._current_month: str = ""

    # ------------------------------------------------------------------
    # 主检查入口
    # ------------------------------------------------------------------

    def check(
        self,
        date: str,
        weights: pd.Series,
        nav_history: Optional[pd.Series] = None,
        daily_pnl: Optional[dict[str, float]] = None,
        industry_map: Optional[dict[str, str]] = None,
    ) -> tuple[pd.Series, list[RiskEvent]]:
        """执行全量风控检查，返回调整后的权重与事件列表。

        Args:
            date: 当前日期。
            weights: 当前持仓权重 Series (stock → weight)。
            nav_history: 组合净值序列（用于组合止损判断）。
            daily_pnl: 当日个股盈亏 (stock → pnl_ratio)。
            industry_map: 股票 → 行业映射。

        Returns:
            (adjusted_weights, events)
        """
        self._events = []
        adjusted = weights.copy()

        # 0. 黑名单过期处理 (每个交易日推进)
        self._day_counter += 1
        expired = [s for s, days in self._blacklist.items() if days <= 0]
        for s in expired:
            del self._blacklist[s]

        # 0.5. 跟踪日度盈亏 (新增)
        if daily_pnl:
            daily_portfolio_pnl = self._track_daily_pnl(date, weights, daily_pnl)

        # 1. 个股止损 (触发时加入黑名单)
        if daily_pnl:
            adjusted = self._check_single_stop(date, adjusted, daily_pnl)
            # 日亏损限额检查 (个股止损之后)
            adjusted = self._check_daily_loss(date, adjusted, daily_pnl)
            # 连续亏损检查
            adjusted = self._check_consecutive_losses(date, adjusted)

        # 2. 组合止损
        if nav_history is not None and len(nav_history) >= 2:
            adjusted = self._check_portfolio_stop(date, adjusted, nav_history)

        # 3. 移动止盈
        adjusted = self._check_trailing_stop(date, adjusted)

        # 4. 仓位限制 (调仓时强制截断 + 等比例再分配)
        adjusted = self._enforce_position_limits(adjusted)

        # 5. 行业集中度
        if industry_map:
            adjusted = self._enforce_sector_limits(date, adjusted, industry_map)

        # 归一化
        total = adjusted.sum()
        if total > 0:
            adjusted = adjusted / total

        # 归一化后再次检查仓位限制
        adjusted = adjusted.clip(upper=self.limits.max_single_position)
        # 限制持仓数量
        positive = adjusted[adjusted > 0]
        if len(positive) > self.limits.max_positions:
            top_stocks = positive.nlargest(self.limits.max_positions).index
            adjusted.loc[adjusted.index.difference(top_stocks)] = 0.0

        return adjusted, self._events

    # ------------------------------------------------------------------
    # 个股止损
    # ------------------------------------------------------------------

    def _check_single_stop(
        self, date: str, weights: pd.Series, daily_pnl: dict[str, float]
    ) -> pd.Series:
        """个股止损：跌幅超过阈值的股票强制清仓 + 加入黑名单。"""
        limits = self.limits
        pnl_s = pd.Series(daily_pnl).reindex(weights.index, fill_value=0)
        triggered = pnl_s <= limits.stop_loss_single

        for stock in pnl_s[triggered].index:
            if weights[stock] > 0:
                weights[stock] = 0.0
                self._events.append(RiskEvent(
                    date=date,
                    event_type="stop_loss_single",
                    detail=f"{stock} 日跌幅 {pnl_s[stock]:.2%} 超过阈值 {limits.stop_loss_single:.2%}",
                    action="liquidate",
                ))
                self._blacklist[stock] = limits.blacklist_days
        return weights

    # ------------------------------------------------------------------
    # 日亏损限额 (新增)
    # ------------------------------------------------------------------

    def _track_daily_pnl(
        self, date: str, weights: pd.Series, daily_pnl: dict[str, float]
    ) -> float:
        """跟踪每日组合盈亏并计算月度累计。"""
        pnl_series = pd.Series(daily_pnl).reindex(weights.index, fill_value=0)
        port_pnl = float((weights * pnl_series).sum())

        self._daily_pnl.append(port_pnl)

        month_key = date[:7]
        if month_key != self._current_month:
            self._monthly_pnl = 0.0
            self._current_month = month_key
        self._monthly_pnl += port_pnl

        return port_pnl

    def _check_daily_loss(
        self, date: str, weights: pd.Series, daily_pnl: dict[str, float]
    ) -> pd.Series:
        """日亏损限额检查。"""
        limits = self.limits
        pnl_series = pd.Series(daily_pnl).reindex(weights.index, fill_value=0)
        port_pnl = float((weights * pnl_series).sum())

        # 日度检查
        if port_pnl <= limits.daily_loss_limit:
            scale = 0.5  # 减仓 50%
            weights = weights * scale
            self._events.append(RiskEvent(
                date=date,
                event_type="daily_loss_limit",
                detail=f"日亏损 {port_pnl:.2%} 超过限额 {limits.daily_loss_limit:.2%}, 减仓至 50%",
                action="reduce_all",
            ))

        # 月度检查
        if self._monthly_pnl <= limits.max_monthly_loss:
            weights = pd.Series(0.0, index=weights.index)
            self._events.append(RiskEvent(
                date=date,
                event_type="monthly_loss_limit",
                detail=f"月度累计亏损 {self._monthly_pnl:.2%} 超过限额 {limits.max_monthly_loss:.2%}, 清仓",
                action="liquidate_all",
            ))

        return weights

    def _check_consecutive_losses(
        self, date: str, weights: pd.Series
    ) -> pd.Series:
        """连续亏损检查。

        若连续 N 天亏损，减仓至 50% 并发出警报。
        """
        limits = self.limits
        if len(self._daily_pnl) == 0:
            return weights

        last_pnl = self._daily_pnl[-1]
        if last_pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self._consecutive_losses >= limits.consecutive_loss_days:
            scale = 0.5
            weights = weights * scale
            self._events.append(RiskEvent(
                date=date,
                event_type="consecutive_losses",
                detail=f"连续 {self._consecutive_losses} 日亏损，减仓至 50%",
                action="reduce_all",
            ))
            self._consecutive_losses = 0  # 重置，避免重复触发

        return weights

    # ------------------------------------------------------------------
    # 组合止损
    # ------------------------------------------------------------------

    def _check_portfolio_stop(
        self, date: str, weights: pd.Series, nav: pd.Series
    ) -> pd.Series:
        """组合止损：日度组合亏损超过阈值 → 全部清仓。"""
        if len(nav) < 2:
            return weights

        daily_loss = float(nav.iloc[-1] / nav.iloc[-2] - 1)
        if daily_loss <= self.limits.stop_loss_portfolio:
            self._events.append(RiskEvent(
                date=date,
                event_type="stop_loss_portfolio",
                detail=f"组合日亏损 {daily_loss:.2%} 超过阈值 {self.limits.stop_loss_portfolio:.2%}",
                action="liquidate_all",
            ))
            return pd.Series(0.0, index=weights.index)

        return weights

    # ------------------------------------------------------------------
    # 移动止盈
    # ------------------------------------------------------------------

    def _check_trailing_stop(
        self, date: str, weights: pd.Series
    ) -> pd.Series:
        """移动止盈：从峰值回撤超过阈值 → 减仓。"""
        for stock in weights.index:
            w = weights[stock]
            if w <= 0:
                continue

            # 更新峰值
            peak = self._peak_weights.get(stock, 0.0)
            if w > peak:
                self._peak_weights[stock] = w
                peak = w

            # 检查回撤
            if peak > 0 and (w - peak) / peak <= self.limits.trailing_stop:
                old_w = w
                weights[stock] = 0.0
                self._events.append(RiskEvent(
                    date=date,
                    event_type="trailing_stop",
                    detail=f"{stock} 从峰值 {peak:.4f} 回撤至 {w:.4f}",
                    action="liquidate",
                ))
                self._blacklist[stock] = self.limits.blacklist_days

        return weights

    # ------------------------------------------------------------------
    # 仓位限制
    # ------------------------------------------------------------------

    def _enforce_position_limits(self, weights: pd.Series) -> pd.Series:
        """强制单只股票仓位不超过上限。

        裁剪超限部分，多余权重转为现金头寸（不重新分配）。
        回测引擎负责处理现金部分。
        """
        limits = self.limits

        # 截断超限仓位
        weights = weights.clip(upper=limits.max_single_position)

        # 持仓数限制: 只保留权重最大的 top N
        positive = weights[weights > 0]
        if len(positive) > limits.max_positions:
            top_stocks = positive.nlargest(limits.max_positions).index
            weights.loc[weights.index.difference(top_stocks)] = 0.0

        return weights

    # ------------------------------------------------------------------
    # 行业集中度
    # ------------------------------------------------------------------

    def _enforce_sector_limits(
        self, date: str, weights: pd.Series, industry_map: dict[str, str]
    ) -> pd.Series:
        """强制行业集中度不超过上限。"""
        limits = self.limits
        # 向量化: groupby 汇总行业权重
        sector_weights = weights.groupby(industry_map).sum()

        # 检查单行业超限
        for sector, sw in sector_weights.items():
            if sw > limits.max_sector_position:
                scale = limits.max_sector_position / sw
                for stock in weights.index:
                    if industry_map.get(stock) == sector:
                        weights[stock] *= scale
                self._events.append(RiskEvent(
                    date=date,
                    event_type="sector_limit",
                    detail=f"行业 {sector} 权重 {sw:.2%} 超过上限 {limits.max_sector_position:.2%}, 缩放 {scale:.2f}",
                    action="reduce",
                ))

        # 检查前3行业集中度
        sorted_sectors = sorted(sector_weights.values, reverse=True)
        top3 = sum(sorted_sectors[:3])
        if top3 > limits.max_top3_sector:
            scale = limits.max_top3_sector / top3
            weights = weights * scale
            self._events.append(RiskEvent(
                date=date,
                event_type="sector_concentration",
                detail=f"前3行业集中度 {top3:.2%} 超过上限 {limits.max_top3_sector:.2%}, 整体缩放 {scale:.2f}",
                action="reduce_all",
            ))

        return weights

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @property
    def events(self) -> list[RiskEvent]:
        """最近一次检查的风控事件。"""
        return self._events

    @property
    def blacklist_stocks(self) -> set[str]:
        """当前黑名单股票集合（用于选股过滤）。"""
        # 递减计数并清理过期
        for s in list(self._blacklist.keys()):
            self._blacklist[s] -= 1
            if self._blacklist[s] <= 0:
                del self._blacklist[s]
        return set(self._blacklist.keys())

    def reset_peaks(self) -> None:
        """重置所有跟踪状态。"""
        self._peak_weights.clear()
        self._blacklist.clear()
        self._day_counter = 0
        self._daily_pnl.clear()
        self._consecutive_losses = 0
        self._monthly_pnl = 0.0
        self._current_month = ""

    # ------------------------------------------------------------------
    # 因子暴露监控 (新增)
    # ------------------------------------------------------------------

    def check_factor_exposure(
        self,
        weights: pd.Series,
        factor_values: dict[str, pd.Series],
    ) -> list[RiskEvent]:
        """检查组合因子暴露是否超限。

        暴露 = Σ(weight_i × factor_value_i)。
        对每个因子计算 z-score，超阈值时报警。

        Args:
            weights: 当前持仓权重 Series。
            factor_values: {因子名 → 各股票的因子值 Series}。

        Returns:
            暴露事件列表。

        数据要求 (生产级):
            - factor_values 中的因子值需已中性化和标准化
            - 因子值需与 weights.index 对齐
        """
        exposure_events = []
        limits = self.limits
        total_exposure = 0.0

        for factor_name, fv in factor_values.items():
            common = weights.index.intersection(fv.dropna().index)
            if len(common) == 0:
                continue

            w = weights.loc[common]
            f = fv.loc[common]
            exposure = float((w * f).sum())
            total_exposure += abs(exposure)

            if abs(exposure) > limits.max_factor_exposure:
                exposure_events.append(RiskEvent(
                    date=str(self._day_counter),
                    event_type="factor_exposure",
                    detail=f"因子 {factor_name} 暴露 {exposure:.2f} 超过上限 {limits.max_factor_exposure}",
                    action="alert",
                ))

        if total_exposure > limits.max_total_exposure:
            exposure_events.append(RiskEvent(
                date=str(self._day_counter),
                event_type="factor_exposure",
                detail=f"总因子暴露 {total_exposure:.2f} 超过上限 {limits.max_total_exposure}",
                action="alert",
            ))

        self._events.extend(exposure_events)
        return exposure_events

    @staticmethod
    def compute_hhi(weights: pd.Series, industry_map: dict[str, str]) -> float:
        """计算 HHI (Herfindahl-Hirschman Index) 行业集中度。

        HHI = Σ(行业权重²), 值越大集中度越高。
        """
        sector_w: dict[str, float] = {}
        for stock, w in weights.items():
            sector = industry_map.get(stock, "未知")
            sector_w[sector] = sector_w.get(sector, 0.0) + w
        return sum(v ** 2 for v in sector_w.values())
