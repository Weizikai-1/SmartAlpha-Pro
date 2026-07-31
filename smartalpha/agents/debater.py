"""多空辩论 Agent - 综合多个 Agent 观点，输出加权决策。

[预留接口] LangGraph workflow 中辩论逻辑通过 llm.debate() 直接调用，
本 Agent 保留为未来独立部署用（如单独评估某只股票的多空分歧）。
"""

from __future__ import annotations

from smartalpha.agents.base import AgentBase, AgentContext, AgentOutput


class DebaterAgent(AgentBase):
    """多空辩论 Agent - 综合 4 个 Agent 输出，产出加权共识。

    不自行调用数据或模型，纯 LLM 推理。
    """

    name = "debater"

    def analyze(self, ctx: AgentContext) -> AgentOutput:
        """综合多个 Agent 输出，辩论后输出加权信号。"""
        if self.llm is None:
            return AgentOutput(
                agent_name=self.name,
                signal="neutral",
                confidence=0.5,
                reasoning="LLM 不可用，无法进行多空辩论",
            )
        # 实际辩论逻辑整合在 graph/workflow.py 的 _stage3_synthesize 中
        return AgentOutput(
            agent_name=self.name,
            signal="neutral",
            confidence=0.5,
            reasoning="多空辩论通过 LangGraph Stage 3 执行",
        )
