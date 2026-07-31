"""基本面分析 Agent — 财务因子 + 行业对标。

调用现有模块:
- core/functions.py → LEVERAGE, DEBT_RATIO, GROWTH, ROIC
- data/industry_fetcher.py → 行业基准
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from smartalpha.agents.base import AgentBase, AgentContext, AgentOutput
from smartalpha.core.functions import FinancialFunctionLibrary

logger = logging.getLogger(__name__)


class FundamentalAgent(AgentBase):
    """基本面分析 Agent。

    分析维度:
    1. 杠杆水平 (LEVERAGE, DEBT_RATIO)
    2. 成长性 (GROWTH)
    3. 盈利能力 (ROIC)
    4. 行业对标
    """

    name = "fundamental"

    def analyze(self, ctx: AgentContext) -> AgentOutput:
        """执行基本面分析。

        流程:
        1. 计算/提取财务因子值
        2. 检索相似历史场景 (ChromaDB)
        3. LLM 推理 → 结构化输出
        """
        similar = self._retrieve_similar(ctx)

        # 从面板数据中计算财务因子
        fin_data = self._compute_financials(ctx)

        # 如果 LLM 可用，使用 LLM 分析
        if self.llm is not None:
            ctx_dict = ctx.to_dict()
            ctx_dict["factor_df"] = fin_data
            result = self.llm.analyze_fundamental(ctx_dict, similar)
            return AgentOutput(
                agent_name=self.name,
                signal=result.get("signal", "neutral"),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                key_metrics=result.get("key_metrics", {}),
                risk_flags=result.get("risk_flags", []),
            )

        # 无 LLM 时的规则兜底
        return self._rule_based_analysis(ctx, fin_data)

    # ------------------------------------------------------------------
    # 财务因子计算
    # ------------------------------------------------------------------

    def _compute_financials(self, ctx: AgentContext) -> pd.DataFrame:
        """从面板数据计算财务相关因子值。"""
        panel = ctx.panel
        if panel is None or panel.empty:
            return pd.DataFrame()

        try:
            lib = FinancialFunctionLibrary()
            price = panel["close"] if "close" in panel.columns else panel.iloc[:, 0]
            volume = panel.get("vol", panel.get("volume", pd.Series(dtype=float)))

            result = {}
            for name in ["LEVERAGE", "DEBT_RATIO", "GROWTH", "ROIC"]:
                try:
                    if lib.has_function(name):
                        if name in ("GROWTH", "ROIC"):
                            val = lib.call(name, price.values)
                        else:
                            val = lib.call(name, price.values)
                        if isinstance(val, np.ndarray) and len(val) > 0:
                            result[name] = val
                except Exception:
                    pass

            if result:
                df = pd.DataFrame(result)
                if hasattr(panel, "index") and hasattr(panel.index, "get_level_values"):
                    df.index = panel.index.get_level_values(0).unique()[-len(df):]
                return df
        except Exception as e:
            logger.warning(f"财务因子计算失败: {e}")

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 规则兜底
    # ------------------------------------------------------------------

    def _rule_based_analysis(
        self, ctx: AgentContext, fin_data: pd.DataFrame
    ) -> AgentOutput:
        """无 LLM 时基于规则的简单分析。"""
        if fin_data.empty:
            return AgentOutput(
                agent_name=self.name,
                signal="neutral",
                confidence=0.5,
                reasoning="无财务数据，无法进行基本面分析",
                key_metrics={},
            )

        reasoning_parts = []
        signal_score = 0.0

        for col in fin_data.columns:
            vals = fin_data[col].dropna()
            if len(vals) == 0:
                continue
            last_val = float(vals.iloc[-1])
            if col == "LEVERAGE":
                if last_val < 2.0:
                    signal_score += 0.15
                    reasoning_parts.append(f"杠杆率 {last_val:.2f} (较低, 财务稳健)")
                elif last_val > 5.0:
                    signal_score -= 0.15
                    reasoning_parts.append(f"杠杆率 {last_val:.2f} (偏高)")
            elif col == "GROWTH":
                if last_val > 0.1:
                    signal_score += 0.2
                    reasoning_parts.append(f"增长率 {last_val:.1%} (高增长)")
                elif last_val < 0:
                    signal_score -= 0.2
                    reasoning_parts.append(f"增长率 {last_val:.1%} (负增长)")
            elif col == "ROIC":
                if last_val > 0.1:
                    signal_score += 0.15
                    reasoning_parts.append(f"ROIC {last_val:.1%} (优秀)")
                elif last_val < 0.05:
                    signal_score -= 0.1
                    reasoning_parts.append(f"ROIC {last_val:.1%} (偏低)")

        signal = "bullish" if signal_score > 0.1 else ("bearish" if signal_score < -0.1 else "neutral")
        confidence = min(abs(signal_score) * 2 + 0.3, 0.85)

        return AgentOutput(
            agent_name=self.name,
            signal=signal,
            confidence=round(confidence, 2),
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "基本面指标中性",
            key_metrics={col: float(fin_data[col].dropna().iloc[-1]) for col in fin_data.columns},
        )
