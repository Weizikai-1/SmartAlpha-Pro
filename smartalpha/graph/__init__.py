"""LangGraph 工作流模块 — TradingAgents 标准 4 阶段 DAG。

State: AgentState (TypedDict, 每键单写, fan-out 友好)
Workflow: build_workflow() 返回 CompiledStateGraph
Entry:    run_analysis(ticker, trade_date, depth) → dict
"""

from smartalpha.graph.state import AgentState
from smartalpha.graph.workflow import build_workflow, run_analysis

__all__ = ["AgentState", "build_workflow", "run_analysis"]
