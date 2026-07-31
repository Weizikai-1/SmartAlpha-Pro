"""Agent 基类 — 所有分析 Agent 的抽象基类。

设计原则:
- 每个 Agent 只做一件事: 接收 AgentContext，返回 AgentOutput
- LLM 调用通过 DeepSeekClient，记忆检索通过 ChromaMemoryStore
- Agent 之间零耦合，完全通过 SharedState 通信
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class AgentContext:
    """传递给每个 Agent 的标准化上下文。

    所有字段均为只读快照，Agent 不应修改。
    """

    stock_code: str
    stock_name: str = ""
    start_date: str = ""
    end_date: str = ""
    analysis_depth: str = "standard"  # quick / standard / deep

    # 数据快照 (由 Stage 1 填充)
    panel: Optional[pd.DataFrame] = None        # MultiIndex(date, stock) 面板
    factor_df: Optional[pd.DataFrame] = None    # 因子值矩阵
    signal_value: Any = None                    # 模型预测信号
    benchmark_returns: Optional[pd.Series] = None
    industry: str = ""
    industry_map: dict[str, str] = field(default_factory=dict)

    # 新闻 Agent 专用
    news_items: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转为可序列化的字典（供 prompt 构建使用）。"""
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "analysis_depth": self.analysis_depth,
            "industry": self.industry,
            "signal_value": self.signal_value,
            "factor_df": self.factor_df,
            "news_items": self.news_items,
        }


@dataclass
class AgentOutput:
    """Agent 统一输出格式。"""

    agent_name: str
    signal: str                     # "bullish" / "bearish" / "neutral"
    confidence: float               # 0.0 ~ 1.0
    reasoning: str                  # LLM 生成的推理文本
    key_metrics: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    tokens_used: int = 0

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_metrics": self.key_metrics,
            "risk_flags": self.risk_flags,
        }


class AgentBase(ABC):
    """所有分析 Agent 的抽象基类。

    子类必须实现:
    - name: Agent 名称 (类属性)
    - analyze(ctx) -> AgentOutput
    """

    name: str = "base"

    def __init__(
        self,
        llm_client: Any = None,
        chroma_store: Any = None,
    ):
        """初始化 Agent。

        Args:
            llm_client: DeepSeekClient 实例（注入，便于测试 mock）。
            chroma_store: ChromaMemoryStore 实例（可选，用于 RAG 增强）。
        """
        self.llm = llm_client
        self.memory = chroma_store

    @abstractmethod
    def analyze(self, ctx: AgentContext) -> AgentOutput:
        """执行分析，返回结构化输出。"""
        ...

    def _retrieve_similar(self, ctx: AgentContext, k: int = 3) -> str:
        """从 ChromaDB 检索相似历史场景，返回格式化文本。"""
        if self.memory is None:
            return ""
        try:
            query = f"{ctx.stock_code} {ctx.industry} {self.name}"
            return self.memory.query_similar(
                collection="market_patterns",
                query_text=query,
                k=k,
            )
        except Exception:
            return ""

    @staticmethod
    def from_context(ctx: AgentContext, name: str) -> AgentContext:
        """创建子集上下文（供子 Agent 使用）。"""
        return ctx
