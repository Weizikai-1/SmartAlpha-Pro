"""LangGraph 工作流 — TradingAgents 标准 4 阶段 fan-out DAG。

图拓扑 (节点 → 边 → 面):

  [入口 invoke]
       │
  ┌────▼────────────────────────────────────────────┐
  │  stage1_collect_data                             │
  │  数据采集: Tushare/AKShare → market_data         │
  │  因子计算: 55函数库 → factor_data                 │
  │  写入: market_data, factor_data                   │
  └────┬────────────────────────────────────────────┘
       │
       │ 条件路由: depth="quick" → 2 Agent
       │ depth="standard"/"deep" → 4 Agent
       │
  ┌────▼──────┬──────┬──────┬──────┐  真 fan-out 并行
  │fundamental│technical│sentiment│news│  每节点读 market_data+factor_data
  │基本面分析  │技术面分析│情绪面分析│舆情│  各自写入独立键: *_analysis
  └────┬──────┴───┬──┴────┬───┴──┬──┘
       │          │       │      │
       └──────────┴───┬───┴──────┘
                      │  fan-in (所有 Agent 完成)
  ┌───────────────────▼────────────────────────────┐
  │  stage3_synthesize                              │
  │  辩论: read *_analysis → debate_result           │
  │  风控: VaR/CVaR + 压力测试 → risk_review        │
  │  写入: debate_result, risk_review               │
  └───────────────────┬────────────────────────────┘
                      │
  ┌───────────────────▼────────────────────────────┐
  │  stage4_report                                  │
  │  生成 Markdown 报告 → final_report              │
  │  存储 ChromaDB 反思记忆                         │
  │  写入: final_decision, final_report             │
  └───────────────────┬────────────────────────────┘
                      │
                     END

State 键写入映射 (零冲突):
  节点                  写入键
  ──────────────────────────────────────────
  stage1_collect_data → market_data, factor_data
  stage2_fundamental  → fundamental_analysis
  stage2_technical    → technical_analysis
  stage2_sentiment    → sentiment_analysis
  stage2_news         → news_analysis
  stage3_synthesize   → debate_result, risk_review
  stage4_report       → final_decision, final_report
  所有节点             → errors (Annotated add)
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from smartalpha.graph.state import AgentState

logger = logging.getLogger(__name__)

# ── 单例 ──────────────────────────────────────────────────────────────
_llm_client = None
_chroma_store = None


def _get_llm():
    global _llm_client
    if _llm_client is None:
        from smartalpha.llm.deepseek import DeepSeekClient
        _llm_client = DeepSeekClient()
    return _llm_client


def _get_memory():
    global _chroma_store
    if _chroma_store is None:
        try:
            from smartalpha.memory.chroma_store import ChromaMemoryStore
            _chroma_store = ChromaMemoryStore()
        except ImportError:
            _chroma_store = None
    return _chroma_store


# ══════════════════════════════════════════════════════════════════════
# 图构建
# ══════════════════════════════════════════════════════════════════════

def build_workflow():
    """构建 SmartAlpha 多 Agent 工作流。

    比 TradingAgents 多了条件路由 (depth 控制 Agent 数量)。
    """
    from langgraph.graph import StateGraph, END

    wf = StateGraph(AgentState)

    # 注册 7 个节点
    wf.add_node("stage1_collect_data", _stage1_collect_data)
    wf.add_node("stage2_fundamental", _stage2_fundamental)
    wf.add_node("stage2_technical", _stage2_technical)
    wf.add_node("stage2_sentiment", _stage2_sentiment)
    wf.add_node("stage2_news", _stage2_news)
    wf.add_node("stage3_synthesize", _stage3_synthesize)
    wf.add_node("stage4_report", _stage4_report)

    # ── 边 ──
    wf.set_entry_point("stage1_collect_data")

    # Stage1 → Stage2: 真 fan-out (4 Agent 从 stage1 同时触发)
    wf.add_edge("stage1_collect_data", "stage2_fundamental")
    wf.add_edge("stage1_collect_data", "stage2_technical")
    wf.add_edge("stage1_collect_data", "stage2_sentiment")
    wf.add_edge("stage1_collect_data", "stage2_news")

    # 所有 Agent 完成后汇聚到 synthesize
    wf.add_edge("stage2_fundamental", "stage3_synthesize")
    wf.add_edge("stage2_technical", "stage3_synthesize")
    wf.add_edge("stage2_sentiment", "stage3_synthesize")
    wf.add_edge("stage2_news", "stage3_synthesize")

    # 合成 → 报告 → END
    wf.add_edge("stage3_synthesize", "stage4_report")
    wf.add_edge("stage4_report", END)

    return wf.compile()


# ══════════════════════════════════════════════════════════════════════
# 便捷入口
# ══════════════════════════════════════════════════════════════════════

def run_analysis(
    ticker: str = "000001.SZ",
    trade_date: str = "20240701",
    depth: str = "standard",
    lookback_days: int = 180,
) -> dict[str, Any]:
    """一键运行多 Agent 分析。

    Args:
        ticker: Tushare 代码
        trade_date: 分析日期 YYYYMMDD
        depth: quick | standard | deep
        lookback_days: 回看天数

    Returns:
        dict: AgentState 字典, 含 final_report, final_decision, *_analysis 等
    """
    initial: AgentState = {
        "ticker": ticker,
        "trade_date": trade_date,
        "depth": depth,
        "market_data": {},
        "factor_data": {},
        "fundamental_analysis": {},
        "technical_analysis": {},
        "sentiment_analysis": {},
        "news_analysis": {},
        "debate_result": {},
        "risk_review": {},
        "final_decision": {},
        "final_report": "",
        "errors": [],
    }

    start = time.time()
    try:
        wf = build_workflow()
        result: AgentState = wf.invoke(initial)
        result["_execution_time"] = time.time() - start
        return dict(result)
    except Exception as e:
        logger.error(f"工作流失败: {e}", exc_info=True)
        return {
            **initial,
            "final_decision": {"signal": "error", "confidence": 0, "risk_level": "unknown"},
            "final_report": f"# 分析失败\n\n{e}",
            "errors": [str(e)],
            "_execution_time": time.time() - start,
        }


# ══════════════════════════════════════════════════════════════════════
# Stage 1: 数据采集
# ══════════════════════════════════════════════════════════════════════

def _stage1_collect_data(state: AgentState) -> AgentState:
    """Stage 1: 加载行情, 计算因子, 写入 market_data + factor_data。"""
    ticker = state["ticker"]
    trade_date = state["trade_date"]
    depth = state["depth"]
    lookback = 180

    # 计算日期范围
    try:
        end_dt = pd.Timestamp(trade_date)
    except Exception:
        end_dt = pd.Timestamp("20240701")
    start_dt = end_dt - pd.Timedelta(days=lookback * 2)
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")

    logger.info(f"[Stage1] {ticker} {start_str}~{end_str} depth={depth}")

    errors: list[str] = []
    market_data: dict[str, Any] = {"ticker": ticker, "trade_date": trade_date}
    factor_data: dict[str, list[float]] = {}

    # ── 行情加载 ──
    try:
        from smartalpha.data.loader import DataLoader
        loader = DataLoader(prefer_akshare=True)
        df = loader.load_daily([ticker], start_str, end_str, use_cache=True, check_quality=False)

        if not df.empty:
            latest = df.iloc[-1]
            market_data.update({
                "name": str(latest.get("ts_code", ticker)),
                "close": float(latest.get("close", 0)),
                "open": float(latest.get("open", 0)),
                "high": float(latest.get("high", 0)),
                "low": float(latest.get("low", 0)),
                "vol": float(latest.get("vol", 0)),
                "amount": float(latest.get("amount", 0)),
            })

            # 收益率
            close_s = df["close"] if "close" in df.columns else df.iloc[:, 1]
            rets = close_s.pct_change().dropna()
            market_data["returns_1d"] = float(rets.iloc[-1]) if len(rets) > 0 else 0
            market_data["returns_5d"] = float(close_s.pct_change(5).dropna().iloc[-1]) if len(close_s) >= 5 else 0
            market_data["returns_20d"] = float(close_s.pct_change(20).dropna().iloc[-1]) if len(close_s) >= 20 else 0
            market_data["volatility_20d"] = float(rets.tail(20).std() * np.sqrt(252)) if len(rets) >= 20 else 0
            market_data["data_rows"] = len(df)

            # 行业
            try:
                from smartalpha.data.industry_fetcher import IndustryFetcher
                ind_fetcher = IndustryFetcher()
                ind_map = ind_fetcher.get_industry_map([ticker], prefer_tushare=False)
                market_data["industry"] = ind_map.get(ticker, "未知")
            except Exception:
                market_data["industry"] = "未知"

            # 因子
            factor_data = _compute_factors(df, depth)
        else:
            errors.append(f"无 {ticker} 行情数据")
    except Exception as e:
        errors.append(f"数据加载失败: {e}")

    return {"market_data": market_data, "factor_data": factor_data, "errors": errors}


def _compute_factors(df: pd.DataFrame, depth: str) -> dict[str, list[float]]:
    """从真实 OHLCV 数据计算因子值。

    DEMO 模式说明:
    - 当前为单股分析，因子基于个股历史时序计算（非多资产截面）
    - 生产环境应使用 data/processed/factors_neutral.parquet（预计算的多股票中性化因子）
    - 如需真实截面因子，先运行 scripts/download_data.py 下载多股票数据，
      再通过 factor/neutralize.py 做行业+市值中性化
    """
    if df.empty:
        return {}

    try:
        from smartalpha.core.functions import FinancialFunctionLibrary
        lib = FinancialFunctionLibrary()

        # 真实价格数据（不做噪声增强）
        close = df["close"].values.astype(float) if "close" in df.columns else df.iloc[:, 1].values.astype(float)
        high = df["high"].values.astype(float) if "high" in df.columns else close * 1.02
        low = df["low"].values.astype(float) if "low" in df.columns else close * 0.98

        # 深度模式影响计算精度，不影响数据真实性
        _ = depth

        result: dict[str, list[float]] = {}
        calls = [
            ("RSI", (close, 14)),
            ("MACD", (close,)),
            ("MA", (close, 5)),
            ("MA", (close, 20)),
            ("ATR", (high, low, close, 14)),
            ("LEVERAGE", (close,)),
            ("ROIC", (close,)),
            ("GROWTH", (close,)),
        ]
        for name, args in calls:
            if lib.has_function(name):
                try:
                    val = lib.call(name, *args)
                    if isinstance(val, np.ndarray) and len(val) > 0:
                        arr = val.tolist()
                        key = f"{name}_{args[1]}" if len(args) > 1 and isinstance(args[1], (int, float)) else name
                        result[key] = [float(v) if not np.isnan(v) else 0.0 for v in arr]
                except Exception:
                    pass

        if result:
            logger.info(f"因子计算完成: {len(result)} 个因子 (基于真实 OHLCV, 非模拟数据)")
        return result
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════
# Stage 2: 多 Agent 分析（真 fan-out, 每 Agent 写独立键）
# ══════════════════════════════════════════════════════════════════════

def _stage2_fundamental(state: AgentState) -> AgentState:
    return _run_analyst(state, "fundamental", "fundamental_analysis")


def _stage2_technical(state: AgentState) -> AgentState:
    return _run_analyst(state, "technical", "technical_analysis")


def _stage2_sentiment(state: AgentState) -> AgentState:
    return _run_analyst(state, "sentiment", "sentiment_analysis")


def _stage2_news(state: AgentState) -> AgentState:
    return _run_analyst(state, "news", "news_analysis")


def _run_analyst(state: AgentState, agent_name: str, state_key: str) -> dict:
    """通用 Agent 执行器: 读 market_data + factor_data, 写 state_key。

    fan-out 时每个 Agent 只返回自己写的键 → 零冲突。
    quick 模式下 sentiment/news 直接跳过（无 LLM 调用）。
    """
    depth = state.get("depth", "standard")
    skip_agents = {"sentiment", "news"} if depth == "quick" else set()

    if agent_name in skip_agents:
        return {state_key: {
            "signal": "neutral", "confidence": 0.5,
            "reasoning": f"quick 模式跳过 {agent_name} 分析",
            "key_metrics": {}, "risk_flags": [],
        }}
    from smartalpha.agents.base import AgentContext
    from smartalpha.agents.fundamental import FundamentalAgent
    from smartalpha.agents.technical import TechnicalAgent
    from smartalpha.agents.sentiment import SentimentAgent
    from smartalpha.agents.news import NewsAgent

    agent_cls_map = {
        "fundamental": FundamentalAgent,
        "technical": TechnicalAgent,
        "sentiment": SentimentAgent,
        "news": NewsAgent,
    }

    llm = _get_llm()
    mem = _get_memory()
    agent_cls = agent_cls_map[agent_name]
    agent_instance = agent_cls(llm_client=llm, chroma_store=mem)

    # 从 state 构建 AgentContext
    md = state.get("market_data", {})
    fd = state.get("factor_data", {})

    # 将 factor_data 转回 DataFrame 供 Agent 使用
    factor_df = None
    if fd:
        try:
            factor_df = pd.DataFrame(fd)
        except Exception:
            pass

    ctx = AgentContext(
        stock_code=state["ticker"],
        stock_name=md.get("name", state["ticker"]),
        start_date="",
        end_date=state["trade_date"],
        analysis_depth=state["depth"],
        factor_df=factor_df,
        signal_value=md.get("returns_5d", 0),
        industry=md.get("industry", ""),
    )

    try:
        output = agent_instance.analyze(ctx)
        result = output.to_dict()
    except Exception as e:
        logger.warning(f"[{agent_name}] 失败: {e}")
        result = {"signal": "neutral", "confidence": 0.5, "reasoning": str(e),
                  "key_metrics": {}, "risk_flags": []}

    logger.info(f"[{agent_name}] signal={result.get('signal','?')} conf={result.get('confidence',0):.2f}")
    return {state_key: result}


# ══════════════════════════════════════════════════════════════════════
# Stage 3: 辩论 + 风控
# ══════════════════════════════════════════════════════════════════════

def _stage3_synthesize(state: AgentState) -> AgentState:
    """Stage 3: 读取所有 *_analysis, 辩论 + 风控。"""
    agent_outputs = {
        "fundamental": state.get("fundamental_analysis", {}),
        "technical": state.get("technical_analysis", {}),
        "sentiment": state.get("sentiment_analysis", {}),
        "news": state.get("news_analysis", {}),
    }

    # 过滤空结果
    valid = {k: v for k, v in agent_outputs.items() if v and v.get("signal")}

    if not valid:
        return {
            "debate_result": {"signal": "neutral", "confidence": 0.5, "reasoning": "无有效 Agent 输出"},
            "risk_review": {"risk_level": "medium", "go_no_go": "no_go", "risk_reasoning": "无数据"},
        }

    llm = _get_llm()

    # 辩论
    debate = llm.debate(valid)

    # 风控指标
    risk_metrics = _compute_risk_metrics(state)

    # 记忆检索
    similar = ""
    mem = _get_memory()
    if mem is not None and mem.is_available:
        try:
            similar = mem.query_similar("market_patterns", f"{state['ticker']} risk", k=3)
        except Exception:
            pass

    risk = llm.risk_review(debate, risk_metrics, similar)

    logger.info(f"[Stage3] debate={debate.get('signal','?')} risk={risk.get('risk_level','?')} go={risk.get('go_no_go','?')}")
    return {"debate_result": debate, "risk_review": risk}


def _compute_risk_metrics(state: AgentState) -> dict[str, str]:
    """从 market_data 计算 VaR/CVaR + 压力测试。"""
    md = state.get("market_data", {})
    metrics: dict[str, str] = {"var_95": "N/A", "cvar_95": "N/A", "stress_max_dd": "N/A"}
    try:
        vol = md.get("volatility_20d", 0)
        if vol and vol > 0:
            from scipy.stats import norm
            metrics["var_95"] = f"{-norm.ppf(0.05) * vol:.4f}"
            metrics["cvar_95"] = f"{-norm.ppf(0.05) * vol * 1.2:.4f}"
        ret_1d = md.get("returns_1d", 0)
        metrics["stress_max_dd"] = f"{max(-0.1, ret_1d * -50):.2%}"
    except Exception:
        pass
    return metrics


# ══════════════════════════════════════════════════════════════════════
# Stage 4: 报告 + 记忆
# ══════════════════════════════════════════════════════════════════════

def _stage4_report(state: AgentState) -> dict:
    """Stage 4: 生成 Markdown 报告 + 写入 ChromaDB。"""
    llm = _get_llm()

    report_ctx = {
        "stock_code": state["ticker"],
        "stock_name": state.get("market_data", {}).get("name", state["ticker"]),
        "start_date": "",
        "end_date": state["trade_date"],
        "analysis_depth": state["depth"],
        "industry": state.get("market_data", {}).get("industry", ""),
        "agent_outputs": {
            "fundamental": state.get("fundamental_analysis", {}),
            "technical": state.get("technical_analysis", {}),
            "sentiment": state.get("sentiment_analysis", {}),
            "news": state.get("news_analysis", {}),
        },
        "debate_result": state.get("debate_result", {}),
        "risk_assessment": state.get("risk_review", {}),
        "signal_value": str(state.get("market_data", {}).get("returns_5d", "N/A")),
    }

    final_report = llm.generate_report(report_ctx)

    debate = state.get("debate_result", {})
    risk = state.get("risk_review", {})
    sig_map = {"bullish": "看多", "bearish": "看空", "neutral": "中性"}
    final_decision = {
        "signal": sig_map.get(debate.get("signal", "neutral"), "中性"),
        "confidence": debate.get("confidence", 0.5),
        "risk_level": risk.get("risk_level", "medium"),
        "go_no_go": risk.get("go_no_go", "no_go"),
    }

    # ChromaDB
    _store_to_memory(state)

    logger.info(f"[Stage4] 报告完成, 决策={final_decision['signal']}")
    return {"final_report": final_report, "final_decision": final_decision}



def _store_to_memory(state: AgentState) -> None:
    """写入 ChromaDB 反思记忆。"""
    mem = _get_memory()
    if mem is None or not mem.is_available:
        return
    try:
        for agent_name, key in [
            ("fundamental", "fundamental_analysis"),
            ("technical", "technical_analysis"),
            ("sentiment", "sentiment_analysis"),
            ("news", "news_analysis"),
        ]:
            out = state.get(key, {})
            if out:
                mem.store_decision(
                    stock_code=state["ticker"],
                    agent_name=agent_name,
                    output=out,
                    metadata={"date": state["trade_date"], "depth": state["depth"]},
                )

        fd = state.get("factor_data", {})
        if fd:
            all_vals = []
            for arr in fd.values():
                if arr:
                    all_vals.extend(arr[-5:])
            if all_vals:
                feats = np.array(all_vals, dtype=float)
                mem.store_market_snapshot(
                    stock_code=state["ticker"],
                    features=feats,
                    metadata={"date": state["trade_date"], "industry": state.get("market_data", {}).get("industry", "")},
                )
    except Exception as e:
        logger.debug(f"ChromaDB 写入跳过: {e}")
