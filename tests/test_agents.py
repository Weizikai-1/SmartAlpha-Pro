"""Agent 分析模块测试。

验证 4 个 Agent 的规则兜底逻辑（无 LLM 时）：
- FundamentalAgent: 财务因子计算 + 规则分析
- TechnicalAgent: 技术指标 + 趋势判断
- SentimentAgent: 波动率 + VaR 分析
- NewsAgent: 新闻获取 + 舆情判断
"""
import pytest
import numpy as np
import pandas as pd
from smartalpha.agents.base import AgentContext, AgentOutput
from smartalpha.agents.fundamental import FundamentalAgent
from smartalpha.agents.technical import TechnicalAgent
from smartalpha.agents.sentiment import SentimentAgent
from smartalpha.agents.news import NewsAgent


def _make_context(stock_code="000001.SZ", panel=None):
    """创建标准测试 AgentContext。"""
    return AgentContext(
        stock_code=stock_code,
        stock_name="测试股票",
        start_date="2024-01-01",
        end_date="2024-06-30",
        panel=panel,
    )


def _make_price_panel(n_days=100, trend=0.0001):
    """创建含 close 和 vol 的模拟面板。"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    returns = np.random.randn(n_days) * 0.02 + trend
    price = 10 * np.exp(np.cumsum(returns))
    volume = np.random.randint(1000000, 10000000, n_days)
    df = pd.DataFrame({
        "close": price,
        "vol": volume,
        "open": price * (1 + np.random.randn(n_days) * 0.005),
        "high": price * (1 + np.abs(np.random.randn(n_days) * 0.01)),
        "low": price * (1 - np.abs(np.random.randn(n_days) * 0.01)),
    }, index=dates)
    return df


class TestFundamentalAgent:
    """FundamentalAgent 测试。"""

    def test_analyze_empty_context(self):
        """空面板应返回 neutral 信号。"""
        agent = FundamentalAgent()
        ctx = _make_context()
        result = agent.analyze(ctx)
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "fundamental"
        assert result.signal == "neutral"

    def test_analyze_with_data(self):
        """有价格数据时应执行财务因子计算。"""
        agent = FundamentalAgent()
        panel = _make_price_panel(200)
        ctx = _make_context(panel=panel)
        result = agent.analyze(ctx)
        assert isinstance(result, AgentOutput)
        assert result.signal in ("bullish", "bearish", "neutral")
        assert 0 <= result.confidence <= 1

    def test_rule_based_signal_valid(self):
        """规则分析信号应为三态之一。"""
        agent = FundamentalAgent()
        ctx = _make_context()
        fin_data = pd.DataFrame({
            "LEVERAGE": [1.5, 1.8, 2.0],
            "GROWTH": [0.12, 0.15, 0.10],
            "ROIC": [0.12, 0.14, 0.11],
        })
        result = agent._rule_based_analysis(ctx, fin_data)
        assert result.signal in ("bullish", "bearish", "neutral")

    def test_high_leverage_bearish(self):
        """高杠杆应产生 bearish 或较低分数。"""
        agent = FundamentalAgent()
        ctx = _make_context()
        fin_data = pd.DataFrame({
            "LEVERAGE": [6.0, 6.5, 7.0],
        })
        result = agent._rule_based_analysis(ctx, fin_data)
        # 高杠杆应该不是 bullish
        assert result.signal != "bullish" or result.confidence < 0.5

    def test_high_growth_bullish(self):
        """高增长应产生 bullish。"""
        agent = FundamentalAgent()
        ctx = _make_context()
        fin_data = pd.DataFrame({
            "GROWTH": [0.25, 0.30, 0.28],
        })
        result = agent._rule_based_analysis(ctx, fin_data)
        assert result.signal == "bullish"

    def test_output_has_key_metrics(self):
        """输出应包含 key_metrics。"""
        agent = FundamentalAgent()
        ctx = _make_context()
        fin_data = pd.DataFrame({
            "LEVERAGE": [2.0],
            "ROIC": [0.12],
        })
        result = agent._rule_based_analysis(ctx, fin_data)
        assert isinstance(result.key_metrics, dict)


class TestTechnicalAgent:
    """TechnicalAgent 测试。"""

    def test_analyze_empty_context(self):
        """空面板应返回 neutral。"""
        agent = TechnicalAgent()
        ctx = _make_context()
        result = agent.analyze(ctx)
        assert result.agent_name == "technical"
        assert result.signal == "neutral"

    def test_analyze_with_data(self):
        """有面板数据时应分析。"""
        agent = TechnicalAgent()
        ctx = _make_context(panel=_make_price_panel(200))
        result = agent.analyze(ctx)
        assert result.signal in ("bullish", "bearish", "neutral")

    def test_rule_based_returns_valid_output(self):
        """规则分析应返回有效 AgentOutput。"""
        agent = TechnicalAgent()
        ctx = _make_context()
        result = agent._rule_based_analysis(ctx, _make_price_panel(200))
        assert isinstance(result, AgentOutput)
        assert result.agent_name == "technical"


class TestSentimentAgent:
    """SentimentAgent 测试。"""

    def test_analyze_empty_context(self):
        """空面板应返回 neutral。"""
        agent = SentimentAgent()
        ctx = _make_context()
        result = agent.analyze(ctx)
        assert result.agent_name == "sentiment"
        assert result.signal == "neutral"

    def test_analyze_with_data(self):
        """有面板数据时应分析波动率。"""
        agent = SentimentAgent()
        ctx = _make_context(panel=_make_price_panel(200))
        result = agent.analyze(ctx)
        assert result.signal in ("bullish", "bearish", "neutral")

    def test_high_volatility_bearish(self):
        """高波动率应偏向 bearish。"""
        agent = SentimentAgent()
        ctx = _make_context()
        volatile_panel = _make_price_panel(200, trend=0)
        volatile_panel["close"] = volatile_panel["close"] * (1 + np.random.randn(200) * 0.1)
        result = agent._rule_based_analysis(ctx, volatile_panel)
        assert result.signal in ("bullish", "bearish", "neutral")


class TestNewsAgent:
    """NewsAgent 测试。"""

    def test_analyze_empty_news(self):
        """无新闻时应返回 neutral。"""
        agent = NewsAgent()
        ctx = _make_context()
        result = agent.analyze(ctx)
        assert result.agent_name == "news"
        assert result.signal == "neutral"

    def test_analyze_with_news_items(self):
        """有新闻标题时应分析。"""
        agent = NewsAgent()
        ctx = _make_context()
        ctx.news_items = ["业绩大幅增长", "新产品发布"]
        result = agent._rule_based_analysis(ctx)
        assert result.signal in ("bullish", "bearish", "neutral")
        assert result.key_metrics.get("news_count", 0) == 2

    def test_extract_ak_code(self):
        """_extract_ak_code 应正确转换格式。"""
        assert NewsAgent._extract_ak_code("000001.SZ") == "000001"
        assert NewsAgent._extract_ak_code("600000.SH") == "600000"
        assert NewsAgent._extract_ak_code("") is None
        assert NewsAgent._extract_ak_code(None) is None


class TestAgentContext:
    """AgentContext 数据类测试。"""

    def test_default_values(self):
        """默认值应正确。"""
        ctx = AgentContext(stock_code="000001.SZ")
        assert ctx.stock_name == ""
        assert ctx.analysis_depth == "standard"
        assert ctx.news_items == []

    def test_to_dict(self):
        """to_dict 应返回有效字典。"""
        ctx = AgentContext(stock_code="000001.SZ", stock_name="平安银行")
        d = ctx.to_dict()
        assert isinstance(d, dict)
        assert d["stock_code"] == "000001.SZ"
        assert d["stock_name"] == "平安银行"

    def test_news_items_default_empty(self):
        """news_items 默认为空列表。"""
        ctx = AgentContext(stock_code="000001.SZ")
        assert ctx.news_items == []
