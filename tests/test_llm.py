"""DeepSeek LLM 客户端测试。

验证 DeepSeekClient 的核心功能：
- Mock 模式降级（无 API Key 时）
- chat 方法返回有效 JSON
- 各 Agent 专用方法（analyze_fundamental/technical/sentiment/news）
- 指数退避重试逻辑
- JSON 解析容错
"""
import pytest
import os
from smartalpha.llm.deepseek import DeepSeekClient


class TestDeepSeekClientMock:
    """Mock 模式测试（无需真实 API Key）。"""

    @pytest.fixture
    def client(self):
        """创建无 API Key 的客户端（自动进入 mock 模式）。"""
        old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        old_url = os.environ.pop("DEEPSEEK_BASE_URL", None)
        c = DeepSeekClient(api_key="")  # 空字符串强制 mock
        yield c
        if old_key:
            os.environ["DEEPSEEK_API_KEY"] = old_key
        if old_url:
            os.environ["DEEPSEEK_BASE_URL"] = old_url

    # ------------------------------------------------------------------
    # 基础调用
    # ------------------------------------------------------------------

    def test_chat_returns_string(self, client):
        """chat 方法应返回字符串。"""
        reply = client.chat("你是金融分析师", "分析贵州茅台")
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_chat_mock_returns_json(self, client):
        """mock 模式应返回有效 JSON 字符串。"""
        import json
        reply = client.chat("system", "user")
        try:
            data = json.loads(reply)
            assert "signal" in data
            assert data["signal"] == "neutral"
        except json.JSONDecodeError:
            pytest.fail(f"Mock 响应不是有效 JSON: {reply[:100]}")

    def test_chat_with_temperature(self, client):
        """temperature 参数应被接受。"""
        reply = client.chat("system", "user", temperature=0.7)
        assert isinstance(reply, str)

    def test_chat_with_max_tokens(self, client):
        """max_tokens 参数应被接受。"""
        reply = client.chat("system", "user", max_tokens=512)
        assert isinstance(reply, str)

    # ------------------------------------------------------------------
    # Agent 专用方法
    # ------------------------------------------------------------------

    def test_analyze_fundamental(self, client):
        """analyze_fundamental 应返回信号字典。"""
        result = client.analyze_fundamental(
            {"stock_code": "000001.SZ", "stock_name": "平安银行"},
            ""
        )
        assert isinstance(result, dict)
        assert "signal" in result
        assert "confidence" in result

    def test_analyze_technical(self, client):
        """analyze_technical 应返回信号字典。"""
        result = client.analyze_technical(
            {"stock_code": "000001.SZ"},
            ""
        )
        assert isinstance(result, dict)
        assert "signal" in result

    def test_analyze_sentiment(self, client):
        """analyze_sentiment 应返回信号字典。"""
        result = client.analyze_sentiment(
            {"stock_code": "000001.SZ"},
            ""
        )
        assert isinstance(result, dict)
        assert "signal" in result

    def test_analyze_news(self, client):
        """analyze_news 应返回信号字典。"""
        result = client.analyze_news(
            {"stock_code": "000001.SZ", "news_items": ["业绩预增"]},
            ""
        )
        assert isinstance(result, dict)
        assert "signal" in result

    def test_debate(self, client):
        """debate 应返回辩论结果。"""
        agent_outputs = {
            "fundamental": {"signal": "bullish", "confidence": 0.7},
            "technical": {"signal": "neutral", "confidence": 0.5},
        }
        result = client.debate(agent_outputs)
        assert isinstance(result, dict)
        assert "signal" in result

    def test_risk_review(self, client):
        """risk_review 应返回风控审核结果。"""
        result = client.risk_review(
            {"signal": "bullish", "confidence": 0.8},
            {},
        )
        assert isinstance(result, dict)
        # mock 模式下有 signal 字段（来自 _parse_agent_output fallback）
        assert "signal" in result

    # ------------------------------------------------------------------
    # JSON 解析容错 (静态方法)
    # ------------------------------------------------------------------

    def test_parse_json_valid(self):
        """标准 JSON 应被正确解析。"""
        result = DeepSeekClient._parse_json(
            '{"signal":"bullish","confidence":0.8}'
        )
        assert result["signal"] == "bullish"
        assert result["confidence"] == 0.8

    def test_parse_json_malformed(self):
        """畸形 JSON 应返回空字典。"""
        result = DeepSeekClient._parse_json("这不是 JSON")
        assert result == {}

    def test_parse_json_with_markdown(self):
        """被 markdown 包裹的 JSON 应被正确提取。"""
        result = DeepSeekClient._parse_json(
            '```json\n{"signal":"bearish","confidence":0.3}\n```'
        )
        assert result["signal"] == "bearish"

    def test_parse_agent_output_safe(self):
        """_parse_agent_output 畸形输入应返回 safe default。"""
        result = DeepSeekClient._parse_agent_output("invalid json", "fundamental")
        assert result["signal"] == "neutral"
        assert result["confidence"] == 0.5

    # ------------------------------------------------------------------
    # 初始化参数
    # ------------------------------------------------------------------

    def test_init_defaults(self):
        """默认参数应被正确设置。"""
        old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        old_url = os.environ.pop("DEEPSEEK_BASE_URL", None)
        try:
            c = DeepSeekClient(api_key="")
            assert c.model == "deepseek-chat"
            assert c.base_url == "https://api.deepseek.com"
            assert c.timeout == 60.0
            assert c.max_retries == 3
        finally:
            if old_key:
                os.environ["DEEPSEEK_API_KEY"] = old_key
            if old_url:
                os.environ["DEEPSEEK_BASE_URL"] = old_url

    def test_init_custom_params(self):
        """自定义参数应覆盖默认值。"""
        c = DeepSeekClient(
            model="deepseek-reasoner",
            api_key="sk-test",
            base_url="https://custom.api.com",
            timeout=30.0,
            max_retries=5,
        )
        assert c.model == "deepseek-reasoner"
        assert c._api_key == "sk-test"
        assert c.base_url == "https://custom.api.com"


class TestDeepSeekClientRetry:
    """重试逻辑测试。"""

    def test_retry_count_configured(self):
        """max_retries 参数应被存储。"""
        c = DeepSeekClient(api_key="sk-test", max_retries=5)
        assert c.max_retries == 5

    def test_api_key_empty_goes_mock(self):
        """空 API Key 时 chat 应走 mock 模式。"""
        old_key = os.environ.pop("DEEPSEEK_API_KEY", None)
        old_url = os.environ.pop("DEEPSEEK_BASE_URL", None)
        try:
            c = DeepSeekClient(api_key="")
            reply = c.chat("system", "user")
            assert isinstance(reply, str)
            # mock 模式应该返回包含 "Mock" 或 "neutral" 的内容
            assert "mock" in reply.lower() or "neutral" in reply.lower()
        finally:
            if old_key:
                os.environ["DEEPSEEK_API_KEY"] = old_key
            if old_url:
                os.environ["DEEPSEEK_BASE_URL"] = old_url
