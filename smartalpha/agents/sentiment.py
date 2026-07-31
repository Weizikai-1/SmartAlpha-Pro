"""市场情绪 Agent — 波动率 + VaR + 涨跌停比例。

调用现有模块:
- risk/var.py → VaR/CVaR 计算
- 自研: 波动率分位数、近期涨跌停比例
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from smartalpha.agents.base import AgentBase, AgentContext, AgentOutput

logger = logging.getLogger(__name__)


class SentimentAgent(AgentBase):
    """市场情绪分析 Agent。

    分析维度:
    1. 波动率分位数 (近期 vs 历史)
    2. VaR/CVaR 风险水平
    3. 涨跌停比例
    4. 趋势强度
    """

    name = "sentiment"

    def analyze(self, ctx: AgentContext) -> AgentOutput:
        """执行市场情绪分析。"""
        similar = self._retrieve_similar(ctx)
        sentiment_data = self._compute_sentiment(ctx)

        if self.llm is not None:
            ctx_dict = ctx.to_dict()
            ctx_dict["factor_df"] = sentiment_data
            result = self.llm.analyze_sentiment(ctx_dict, similar)
            return AgentOutput(
                agent_name=self.name,
                signal=result.get("signal", "neutral"),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                key_metrics=result.get("key_metrics", {}),
                risk_flags=result.get("risk_flags", []),
            )

        return self._rule_based_analysis(ctx, sentiment_data)

    # ------------------------------------------------------------------
    # 情绪指标计算
    # ------------------------------------------------------------------

    def _compute_sentiment(self, ctx: AgentContext) -> pd.DataFrame:
        """计算情绪相关指标。"""
        panel = ctx.panel
        if panel is None or panel.empty:
            return pd.DataFrame()

        try:
            close = panel["close"].values if "close" in panel.columns else panel.iloc[:, 0].values
            returns = np.diff(np.log(close.clip(min=0.01)))
            n = len(returns)

            if n < 20:
                return pd.DataFrame()

            # 波动率分位数
            vol_20 = np.std(returns[-20:]) * np.sqrt(252) if n >= 20 else np.nan
            vol_60 = np.std(returns[-60:]) * np.sqrt(252) if n >= 60 else np.nan
            vol_all = np.std(returns) * np.sqrt(252)
            vol_percentile = vol_20 / max(vol_all, 1e-10) if vol_all > 0 else 1.0

            # VaR (简单参数法)
            from smartalpha.risk.var import VaRCalculator
            ret_series = pd.Series(returns[-252:] if n >= 252 else returns)
            var_result = VaRCalculator.parametric(ret_series)

            # 涨跌停比例
            limit_ratio = 0.0
            if n >= 20:
                limit_up = np.sum(returns[-20:] > 0.09)
                limit_down = np.sum(returns[-20:] < -0.09)
                limit_ratio = (limit_up - limit_down) / 20

            # 趋势强度 (近期收益 vs 年化波动)
            recent_ret = np.mean(returns[-10:]) if n >= 10 else 0
            trend_strength = recent_ret / max(np.std(returns[-60:]), 1e-10) if n >= 60 else 0

            data = {
                "vol_20d_annual": [vol_20],
                "vol_60d_annual": [vol_60],
                "vol_percentile": [vol_percentile],
                "var_95": [var_result.var_95],
                "cvar_95": [var_result.cvar_95],
                "limit_ratio_20d": [limit_ratio],
                "trend_strength": [trend_strength],
            }
            return pd.DataFrame(data)
        except Exception as e:
            logger.warning(f"情绪指标计算失败: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # 规则兜底
    # ------------------------------------------------------------------

    def _rule_based_analysis(
        self, ctx: AgentContext, sentiment_data: pd.DataFrame
    ) -> AgentOutput:
        """无 LLM 时基于规则的简单分析。"""
        if sentiment_data.empty:
            return AgentOutput(
                agent_name=self.name,
                signal="neutral",
                confidence=0.5,
                reasoning="无情绪指标数据",
                key_metrics={},
            )

        row = sentiment_data.iloc[0]
        reasoning_parts = []
        signal_score = 0.0

        vol_pct = row.get("vol_percentile", 1.0)
        if vol_pct > 1.5:
            signal_score -= 0.15
            reasoning_parts.append(f"波动率处于高位 ({vol_pct:.1%}分位)")
        elif vol_pct < 0.7:
            signal_score += 0.1
            reasoning_parts.append(f"波动率处于低位 ({vol_pct:.1%}分位)")

        limit_r = row.get("limit_ratio_20d", 0)
        if limit_r > 0.05:
            signal_score += 0.15
            reasoning_parts.append("近期涨停多于跌停,情绪偏多")
        elif limit_r < -0.05:
            signal_score -= 0.15
            reasoning_parts.append("近期跌停多于涨停,情绪偏空")

        trend = row.get("trend_strength", 0)
        if trend > 0.5:
            signal_score += 0.1
        elif trend < -0.5:
            signal_score -= 0.1

        signal = "bullish" if signal_score > 0.1 else ("bearish" if signal_score < -0.1 else "neutral")
        confidence = min(abs(signal_score) * 2 + 0.3, 0.85)

        return AgentOutput(
            agent_name=self.name,
            signal=signal,
            confidence=round(confidence, 2),
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "市场情绪中性",
            key_metrics={k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.floating))},
        )
