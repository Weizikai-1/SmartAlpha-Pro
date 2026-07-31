"""多智能体模块 — LangGraph 节点实现。

提供:
- AgentBase / AgentContext / AgentOutput: 基础设施
- FundamentalAgent: 基本面分析
- TechnicalAgent: 技术面分析
- SentimentAgent: 市场情绪分析
- NewsAgent: 新闻舆情分析
- DebaterAgent: 多空辩论合成
- RiskReviewerAgent: 风控审核
"""

from smartalpha.agents.base import AgentBase, AgentContext, AgentOutput
from smartalpha.agents.fundamental import FundamentalAgent
from smartalpha.agents.technical import TechnicalAgent
from smartalpha.agents.sentiment import SentimentAgent
from smartalpha.agents.news import NewsAgent
from smartalpha.agents.debater import DebaterAgent
from smartalpha.agents.risk_reviewer import RiskReviewerAgent

__all__ = [
    "AgentBase", "AgentContext", "AgentOutput",
    "FundamentalAgent", "TechnicalAgent",
    "SentimentAgent", "NewsAgent",
    "DebaterAgent", "RiskReviewerAgent",
]
