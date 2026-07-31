"""风控审查 Agent - 审核投资建议的风险，给出 go/no-go 决策。

调用现有模块:
- risk/manager.py -> RiskManager.check()
- risk/stress.py -> StressTester.run_all()
- risk/var.py -> VaRCalculator

[预留接口] LangGraph workflow 中风控通过 llm.risk_review() 直接调用，
本 Agent 保留为独立风控评估场景使用（如批量审核持仓组合）。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from smartalpha.agents.base import AgentBase, AgentContext, AgentOutput

logger = logging.getLogger(__name__)


class RiskReviewerAgent(AgentBase):
    """风控审查 Agent。

    分析维度:
    1. VaR/CVaR 风险水平
    2. 压力测试最大回撤
    3. 日亏损限额状态
    4. LLM 综合审核
    """

    name = "risk_reviewer"

    def analyze(self, ctx: AgentContext) -> AgentOutput:
        """执行风控审查。

        基于现有 risk/ 模块计算量化指标，LLM 做最终审核。
        """
        risk_metrics = self._compute_risk_metrics(ctx)
        debate_result = {
            "signal": "neutral",
            "confidence": 0.5,
            "reasoning": "无辩论结果",
        }

        if self.llm is not None:
            similar = self._retrieve_similar(ctx)
            result = self.llm.risk_review(debate_result, risk_metrics, similar)
            return AgentOutput(
                agent_name=self.name,
                signal="neutral",
                confidence=1.0,
                reasoning=result.get("risk_reasoning", ""),
                key_metrics={
                    "risk_level": result.get("risk_level", "medium"),
                    "go_no_go": result.get("go_no_go", "no_go"),
                    **risk_metrics,
                },
            )

        return self._rule_based_review(risk_metrics)

    # ------------------------------------------------------------------
    # 风险指标计算
    # ------------------------------------------------------------------

    def _compute_risk_metrics(self, ctx: AgentContext) -> dict:
        """计算量化风控指标。"""
        panel = ctx.panel
        metrics = {
            "var_95": "N/A",
            "cvar_95": "N/A",
            "stress_max_dd": "N/A",
            "daily_loss_ok": "ok",
        }
        if panel is None or panel.empty:
            return metrics

        try:
            close = panel["close"] if "close" in panel.columns else panel.iloc[:, 0]
            rets = close.pct_change().dropna()
            if len(rets) < 20:
                return metrics

            from smartalpha.risk.var import VaRCalculator
            vr = VaRCalculator.parametric(rets)
            metrics["var_95"] = f"{vr.var_95:.4f}"
            metrics["cvar_95"] = f"{vr.cvar_95:.4f}"

            from smartalpha.risk.stress import StressTester
            results = StressTester.run_all(rets)
            if results:
                metrics["stress_max_dd"] = f"{min(r.max_drawdown for r in results):.2%}"
        except Exception as e:
            logger.warning(f"风控指标计算失败: {e}")

        return metrics

    # ------------------------------------------------------------------
    # 规则兜底
    # ------------------------------------------------------------------

    def _rule_based_review(self, metrics: dict) -> AgentOutput:
        """无 LLM 时基于规则的简单审核。"""
        risk_level = "medium"
        go_no_go = "no_go"
        reasons = []

        try:
            var_95 = float(metrics.get("var_95", "0"))
            if var_95 > -0.01:
                risk_level = "low"
                go_no_go = "go"
            elif var_95 > -0.03:
                risk_level = "medium"
                go_no_go = "go"
            elif var_95 < -0.05:
                risk_level = "high"
                reasons.append(f"VaR95={var_95:.2%} 过高")
        except (ValueError, TypeError):
            pass

        return AgentOutput(
            agent_name=self.name,
            signal="neutral",
            confidence=1.0,
            reasoning="; ".join(reasons) if reasons else "基于VaR阈值的规则审核",
            key_metrics={"risk_level": risk_level, "go_no_go": go_no_go, **metrics},
        )
