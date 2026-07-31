"""LangGraph AgentState — 遵循 TradingAgents 设计标准。

设计原则:
1. TypedDict (非 Pydantic) — LangGraph 原生支持，序列化友好
2. 每键单写 — 并行 fan-out 时各 Agent 写独立键，无需 Reducer 合并
3. 最小冗余 — 不在 State 中放 DataFrame，用 dict 表示结构化数据
4. 分组注释 — [入口] [数据] [分析] [综合] [报告] [元信息] 六段清晰

节点 → 键 映射:
  entry                  → ticker, trade_date, depth
  stage1_collect_data    → market_data, factor_data
  stage2_fundamental     → fundamental_analysis
  stage2_technical       → technical_analysis
  stage2_sentiment       → sentiment_analysis
  stage2_news            → news_analysis
  stage3_synthesize      → debate_result, risk_review
  stage4_report          → final_decision, final_report
  所有节点               → errors (Annotated add)
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict):
    """SmartAlpha 多 Agent 分析共享状态。

    ── 图拓扑 ──
                    ┌─ fundamental_analysis ─┐
    [数据采集] ────┼─ technical_analysis  ───┼── [辩论+风控] ── [报告生成]
                    ├─ sentiment_analysis ───┤
                    └─ news_analysis ────────┘
    quick 模式: 只跑 fundamental + technical
    """

    # ══════════════════════════════════════════════════════════════════
    # [入口] 用户输入 — 由 invoke 设置，节点不写
    # ══════════════════════════════════════════════════════════════════

    ticker: str
    """Tushare 代码, 如 000001.SZ"""

    trade_date: str
    """分析日期 YYYYMMDD, 默认 20240701"""

    depth: str
    """分析深度: quick(2Agent) | standard(4Agent) | deep(4Agent+大截面)"""

    # ══════════════════════════════════════════════════════════════════
    # [数据] 由 stage1 写入 — 无 LLM, 纯计算
    # ══════════════════════════════════════════════════════════════════

    market_data: dict[str, Any]
    """行情快照: {ticker, name, industry, close, high, low, vol, amount, returns_1d, returns_5d, ...}"""

    factor_data: dict[str, list[float]]
    """因子值: {RSI: [...], MACD: [...], MA_20: [...], LEVERAGE: [...], ROIC: [...], GROWTH: [...]}"""

    # ══════════════════════════════════════════════════════════════════
    # [分析] 由 Stage 2 各 Agent 独立写入 — 真 fan-out 并行, 每键单写
    # ══════════════════════════════════════════════════════════════════

    fundamental_analysis: dict[str, Any]
    """基本面 Agent 输出: {signal, confidence, reasoning, key_metrics, risk_flags}"""

    technical_analysis: dict[str, Any]
    """技术面 Agent 输出: {signal, confidence, reasoning, key_metrics, risk_flags}"""

    sentiment_analysis: dict[str, Any]
    """情绪面 Agent 输出: {signal, confidence, reasoning, key_metrics, risk_flags}"""

    news_analysis: dict[str, Any]
    """舆情面 Agent 输出: {signal, confidence, reasoning, key_metrics, risk_flags}"""

    # ══════════════════════════════════════════════════════════════════
    # [综合] 由 stage3 写入 — 辩论 + 风控
    # ══════════════════════════════════════════════════════════════════

    debate_result: dict[str, Any]
    """多空辩论结果: {signal, confidence, reasoning, dissenting_agents, consensus_level}"""

    risk_review: dict[str, Any]
    """风控审核: {risk_level, go_no_go, risk_reasoning, var_95, cvar_95, stress_max_dd}"""

    # ══════════════════════════════════════════════════════════════════
    # [报告] 由 stage4 写入 — 最终产出
    # ══════════════════════════════════════════════════════════════════

    final_decision: dict[str, Any]
    """最终决策: {signal, confidence, risk_level}"""

    final_report: str
    """Markdown 格式完整分析报告"""

    # ══════════════════════════════════════════════════════════════════
    # [元信息] 所有节点追加写入 — Annotated Reducer
    # ══════════════════════════════════════════════════════════════════

    errors: Annotated[list[str], add]
    """所有阶段的错误信息, operator.add 自动拼接"""
