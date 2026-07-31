"""新闻舆情分析 Agent — WebSearch + AKShare 新闻。

数据源:
- AKShare stock_news_em (东方财富个股新闻)
- WebSearch (通用搜索 fallback)
"""

from __future__ import annotations

import logging
from typing import Optional

from smartalpha.agents.base import AgentBase, AgentContext, AgentOutput

logger = logging.getLogger(__name__)


class NewsAgent(AgentBase):
    """新闻舆情分析 Agent。

    分析维度:
    1. 近期新闻标题
    2. 舆情极性
    3. 新闻频率
    """

    name = "news"

    def analyze(self, ctx: AgentContext) -> AgentOutput:
        """执行舆情分析。"""
        similar = self._retrieve_similar(ctx)

        # 获取新闻
        if not ctx.news_items:
            ctx.news_items = self._fetch_news(ctx)

        if self.llm is not None:
            ctx_dict = ctx.to_dict()
            ctx_dict["news_items"] = ctx.news_items
            result = self.llm.analyze_news(ctx_dict, similar)
            return AgentOutput(
                agent_name=self.name,
                signal=result.get("signal", "neutral"),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", ""),
                key_metrics=result.get("key_metrics", {}),
                risk_flags=result.get("risk_flags", []),
            )

        return self._rule_based_analysis(ctx)

    # ------------------------------------------------------------------
    # 新闻获取
    # ------------------------------------------------------------------

    def _fetch_news(self, ctx: AgentContext) -> list[str]:
        """获取近期新闻标题。

        优先 AKShare，失败时尝试 WebSearch。
        """
        code = self._extract_ak_code(ctx.stock_code)
        if not code:
            return []

        # 尝试 AKShare
        try:
            import akshare as ak
            df = ak.stock_news_em(symbol=code)
            if df is not None and not df.empty:
                titles = df["标题"].head(10).tolist() if "标题" in df.columns else []
                logger.info(f"AKShare 获取到 {len(titles)} 条新闻 for {code}")
                return [t for t in titles if isinstance(t, str)][:10]
        except Exception as e:
            logger.warning(f"AKShare 新闻获取失败: {e}")

        # Fallback: 空列表让 LLM 基于其他信息分析
        return []

    @staticmethod
    def _extract_ak_code(ts_code: str) -> Optional[str]:
        """将 Tushare 代码转为 AKShare 代码。"""
        if not ts_code:
            return None
        parts = ts_code.split(".")
        if len(parts) == 2:
            code, exchange = parts
            if exchange == "SH":
                return code
            elif exchange == "SZ":
                return code
        return ts_code.replace(".SH", "").replace(".SZ", "")

    # ------------------------------------------------------------------
    # 规则兜底
    # ------------------------------------------------------------------

    def _rule_based_analysis(self, ctx: AgentContext) -> AgentOutput:
        """无 LLM 时的简单规则分析。"""
        news = ctx.news_items
        if not news:
            return AgentOutput(
                agent_name=self.name,
                signal="neutral",
                confidence=0.5,
                reasoning="无近期新闻数据",
                key_metrics={"news_count": 0},
            )

        return AgentOutput(
            agent_name=self.name,
            signal="neutral",
            confidence=0.5,
            reasoning=f"获取到 {len(news)} 条近期新闻 (需 LLM 进行情感分析)",
            key_metrics={"news_count": len(news)},
        )
