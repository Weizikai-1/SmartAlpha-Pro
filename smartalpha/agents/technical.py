"""技术面分析 Agent — RSI/MACD/均线/布林带等指标。

调用现有模块:
- core/functions.py → RSI, MACD, MA, BOLL, KDJ, ATR
- core/executor.py → 表达式求值
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from smartalpha.agents.base import AgentBase, AgentContext, AgentOutput
from smartalpha.core.functions import FinancialFunctionLibrary

logger = logging.getLogger(__name__)


class TechnicalAgent(AgentBase):
    """技术面分析 Agent。

    分析维度:
    1. 趋势类: MA(5/10/20/60), MACD
    2. 反转类: RSI, KDJ
    3. 波动类: BOLL, ATR
    """

    name = "technical"

    def analyze(self, ctx: AgentContext) -> AgentOutput:
        """执行技术面分析。"""
        similar = self._retrieve_similar(ctx)
        tech_data = self._compute_technicals(ctx)

        if self.llm is not None:
            ctx_dict = ctx.to_dict()
            ctx_dict["factor_df"] = tech_data
            result = self.llm.analyze_technical(ctx_dict, similar)
            return AgentOutput(
                agent_name=self.name,
                signal=result.get("signal", "neutral"),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                key_metrics=result.get("key_metrics", {}),
                risk_flags=result.get("risk_flags", []),
            )

        return self._rule_based_analysis(ctx, tech_data)

    # ------------------------------------------------------------------
    # 技术指标计算
    # ------------------------------------------------------------------

    def _compute_technicals(self, ctx: AgentContext) -> pd.DataFrame:
        """计算技术指标因子值。"""
        panel = ctx.panel
        if panel is None or panel.empty:
            return pd.DataFrame()

        try:
            close = panel["close"].values if "close" in panel.columns else panel.iloc[:, 0].values
            high = panel["high"].values if "high" in panel.columns else close * 1.02
            low = panel["low"].values if "low" in panel.columns else close * 0.98
            volume = panel.get("vol", panel.get("volume", np.ones_like(close))).values

            lib = FinancialFunctionLibrary()
            result = {}
            indicators = {
                "RSI_14": ("RSI", [close, 14]),
                "MA_5": ("MA", [close, 5]),
                "MA_20": ("MA", [close, 20]),
                "MA_60": ("MA", [close, 60]),
                "ATR_14": ("ATR", [high, low, close, 14]),
                "BOLL_UPPER": ("BOLL", [close, 20, 2]),
                "MACD": ("MACD", [close]),
            }

            for col_name, (func_name, args) in indicators.items():
                try:
                    val = lib.call(func_name, *args)
                    if isinstance(val, np.ndarray) and len(val) > 0:
                        result[col_name] = val
                except Exception:
                    pass

            if result:
                df = pd.DataFrame(result)
                if hasattr(panel, "index") and hasattr(panel.index, "get_level_values"):
                    df.index = panel.index.get_level_values(0).unique()[-len(df):]
                return df
        except Exception as e:
            logger.warning(f"技术指标计算失败: {e}")

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 规则兜底
    # ------------------------------------------------------------------

    def _rule_based_analysis(
        self, ctx: AgentContext, tech_data: pd.DataFrame
    ) -> AgentOutput:
        """无 LLM 时基于规则的简单分析。"""
        if tech_data.empty:
            return AgentOutput(
                agent_name=self.name,
                signal="neutral",
                confidence=0.5,
                reasoning="无技术指标数据",
                key_metrics={},
            )

        reasoning_parts = []
        signal_score = 0.0
        metrics = {}

        for col in tech_data.columns:
            vals = tech_data[col].dropna()
            if len(vals) < 2:
                continue
            last = float(vals.iloc[-1])
            prev = float(vals.iloc[-2])
            metrics[col] = last

            if "RSI" in col:
                if last < 30:
                    signal_score += 0.2
                    reasoning_parts.append(f"RSI({last:.1f}) 超卖")
                elif last > 70:
                    signal_score -= 0.2
                    reasoning_parts.append(f"RSI({last:.1f}) 超买")
            elif col == "MA_5" and "MA_20" in tech_data.columns:
                ma20 = float(tech_data["MA_20"].dropna().iloc[-1])
                if last > ma20:
                    signal_score += 0.1
                else:
                    signal_score -= 0.1
            elif col == "MACD" and "MA_5" in tech_data.columns:
                # 简化的 MACD 解读
                pass

        signal = "bullish" if signal_score > 0.1 else ("bearish" if signal_score < -0.1 else "neutral")
        confidence = min(abs(signal_score) * 2 + 0.3, 0.85)

        return AgentOutput(
            agent_name=self.name,
            signal=signal,
            confidence=round(confidence, 2),
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "技术指标中性",
            key_metrics=metrics,
        )
