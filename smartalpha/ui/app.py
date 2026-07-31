"""SmartAlpha Pro — Streamlit 多 Agent 量化分析平台 (TradingAgents 标准)。"""

import streamlit as st

st.set_page_config(
    page_title="SmartAlpha Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 SmartAlpha Pro")
st.caption("LangGraph 多智能体协作 · A 股智能分析")

# ── Sidebar ──
with st.sidebar:
    st.header("分析参数")
    ticker = st.text_input("股票代码", "000001.SZ", help="Tushare 格式")
    trade_date = st.date_input("分析日期").strftime("%Y%m%d") if st.date_input("分析日期") else "20240701"
    depth = st.selectbox("分析深度", ["quick", "standard", "deep"],
                         help="quick=2Agent | standard=4Agent | deep=4Agent+大截面")
    analyze_btn = st.button("开始分析", type="primary", use_container_width=True)
    st.divider()
    st.caption("SmartAlpha Pro v2.0 · LangGraph + DeepSeek")

# ── Main ──
if analyze_btn:
    with st.spinner(f"分析中 ({depth} 模式)..."):
        from smartalpha.graph import run_analysis
        result = run_analysis(ticker=ticker, trade_date=trade_date, depth=depth)

    st.divider()

    # ── 决策卡片 ──
    decision = result.get("final_decision", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sig = decision.get("signal", "?")
        icon = {"看多": "🟢", "看空": "🔴", "中性": "🟡"}.get(sig, "⬜")
        st.metric("决策", f"{icon} {sig}")
    with c2:
        st.metric("置信度", f"{decision.get('confidence', 0):.0%}")
    with c3:
        st.metric("风险", decision.get("risk_level", "?"))
    with c4:
        st.metric("耗时", f"{result.get('_execution_time', 0):.1f}s")

    # ── Agent 面板 ──
    st.subheader("Agent 分析")
    agent_slots = [
        ("fundamental_analysis", "📈 基本面"),
        ("technical_analysis", "📉 技术面"),
        ("sentiment_analysis", "📊 情绪面"),
        ("news_analysis", "📰 舆情面"),
    ]
    icons = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}
    active = [(k, n) for k, n in agent_slots if result.get(k)]
    if active:
        cols = st.columns(len(active))
        for i, (key, name) in enumerate(active):
            out = result[key]
            sig = out.get("signal", "neutral")
            with cols[i]:
                st.metric(f"{icons.get(sig, '⬜')} {name}", f"{out.get('confidence', 0):.0%}")
                with st.expander("详情"):
                    st.write(out.get("reasoning", "—"))
                    if out.get("key_metrics"):
                        st.json(out["key_metrics"])
    else:
        st.info("无 Agent 输出")

    # ── 辩论 + 风控 ──
    debate = result.get("debate_result", {})
    risk = result.get("risk_review", {})
    if debate or risk:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("多空辩论")
            st.metric("综合信号", debate.get("signal", "?"))
            st.write(debate.get("reasoning", ""))
        with col_b:
            st.subheader("风控审核")
            go = risk.get("go_no_go", "?")
            if go == "go":
                st.success("审核通过")
            else:
                st.error(f"审核: {go}")
            st.write(risk.get("risk_reasoning", ""))

    # ── 报告 ──
    report = result.get("final_report", "")
    if report:
        with st.expander("完整报告", expanded=True):
            st.markdown(report)

    # ── 错误 ──
    if result.get("errors"):
        st.divider()
        with st.expander("诊断信息"):
            for e in result["errors"]:
                st.text(f"· {e}")

else:
    st.markdown("""
    ### 功能特性
    - **LangGraph 多 Agent**: 4 个专业 Agent 并行分析，辩论 + 风控审核
    - **DeepSeek LLM**: 结构化分析报告 + ChromaDB 反思记忆
    - **55 因子表达式引擎**: 词法分析→语法解析→AST 执行
    - **A 股真实费率**: 佣金万三+印花税千0.5+滑点千1+冲击成本

    ### 使用
    1. 左侧输入股票代码和日期
    2. 选择分析深度
    3. 点击「开始分析」

    ```bash
    streamlit run smartalpha/ui/app.py
    ```
    """)
    st.caption("SmartAlpha Pro · 量化分析不构成投资建议")
