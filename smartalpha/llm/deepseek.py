"""DeepSeek API 统一封装。

通过 OpenAI 兼容接口调用 DeepSeek，为所有 Agent 提供 LLM 能力。
每个 Agent 有独立的 prompt 模板，确保输出结构化可解析。

使用:
    client = DeepSeekClient()
    result = client.analyze_fundamental(ctx, similar_cases)
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── JSON 解析 prompt 后缀 ──────────────────────────────────────────────
_JSON_SUFFIX = """
请严格按照以下 JSON 格式输出（不要包含其他文字）:
{"signal": "bullish|bearish|neutral", "confidence": 0.0~1.0, "reasoning": "分析依据", "key_metrics": {}, "risk_flags": []}
"""

_DEBATE_JSON_SUFFIX = """
请严格按照以下 JSON 格式输出:
{"signal": "bullish|bearish|neutral", "confidence": 0.0~1.0, "reasoning": "综合判断", "dissenting_agents": [], "consensus_level": "high|medium|low"}
"""

_RISK_JSON_SUFFIX = """
请严格按照以下 JSON 格式输出:
{"risk_level": "low|medium|high|extreme", "go_no_go": "go|no_go", "risk_reasoning": "风控审核意见"}
"""


class DeepSeekClient:
    """DeepSeek API 客户端 — 通过 OpenAI SDK 调用。

    密钥优先级:
    1. 环境变量 DEEPSEEK_API_KEY
    2. .env 文件中的 DEEPSEEK_API_KEY

    使用示例:
        client = DeepSeekClient(model="deepseek-chat")
        reply = client.chat("你是金融分析师", "分析贵州茅台")
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.timeout = timeout
        self.max_retries = max_retries
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self._client = None  # 延迟初始化

    # ------------------------------------------------------------------
    # 基础调用
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """发送对话请求，返回文本回复。

        无 API Key 或 openai 未安装时自动降级为 mock 模式。
        """
        if not self._api_key:
            return self._mock_response(system_prompt, user_prompt)

        if not self._ensure_client():
            return self._mock_response(system_prompt, user_prompt)

        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                wait = min(2 ** attempt, 10)
                logger.warning(f"DeepSeek API 第 {attempt + 1} 次重试: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
        logger.error("DeepSeek API 所有重试均失败，降级为 mock")
        return self._mock_response(system_prompt, user_prompt)

    def _ensure_client(self) -> bool:
        """延迟初始化 OpenAI 客户端。返回 False 表示不可用。"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except ImportError:
                logger.warning("openai 未安装 — DeepSeek 降级为 mock 模式")
                self._client = None
                return False
        return self._client is not None

    def _mock_response(self, system: str, user: str) -> str:
        """无 API Key 时的 mock 响应（用于测试和开发）。"""
        logger.info("DeepSeek mock 模式 (未配置 DEEPSEEK_API_KEY)")
        return (
            '{"signal":"neutral","confidence":0.5,'
            '"reasoning":"Mock 模式: 未配置 DEEPSEEK_API_KEY，请设置后获得真实分析",'
            '"key_metrics":{},"risk_flags":["mock_mode"]}'
        )

    # ------------------------------------------------------------------
    # Agent 专用 prompt 模板
    # ------------------------------------------------------------------

    AGENT_SYSTEM_PROMPTS = {
        "fundamental": "你是一位资深 A 股基本面分析师，专注于财务数据分析和价值评估。",
        "technical": "你是一位 A 股技术分析师，精通 K 线形态、技术指标和多空信号判断。",
        "sentiment": "你是一位市场情绪分析师，擅长从波动率、资金流向和风险指标中判断市场情绪。",
        "news": "你是一位财经新闻舆情分析师，能够从新闻标题和摘要中提取市场情绪和潜在影响。",
    }

    def analyze_fundamental(self, ctx_dict: dict, similar_cases: str = "") -> dict:
        """基本面分析。"""
        return self._agent_analyze("fundamental", ctx_dict, similar_cases)

    def analyze_technical(self, ctx_dict: dict, similar_cases: str = "") -> dict:
        """技术面分析。"""
        return self._agent_analyze("technical", ctx_dict, similar_cases)

    def analyze_sentiment(self, ctx_dict: dict, similar_cases: str = "") -> dict:
        """市场情绪分析。"""
        return self._agent_analyze("sentiment", ctx_dict, similar_cases)

    def analyze_news(self, ctx_dict: dict, similar_cases: str = "") -> dict:
        """新闻舆情分析。"""
        return self._agent_analyze("news", ctx_dict, similar_cases)

    def _agent_analyze(
        self, agent_type: str, ctx_dict: dict, similar_cases: str
    ) -> dict:
        """通用 Agent 分析调用 — 发送 prompt 并解析 JSON。"""
        system = self.AGENT_SYSTEM_PROMPTS[agent_type]
        user = self._build_agent_user_prompt(agent_type, ctx_dict, similar_cases)
        raw = self.chat(system, user)
        return self._parse_agent_output(raw, agent_type)

    def _build_agent_user_prompt(
        self, agent_type: str, ctx: dict, similar: str
    ) -> str:
        """构建 Agent 专用的 user prompt。"""
        stock_info = (
            f"股票: {ctx.get('stock_name', '')} ({ctx.get('stock_code', '')})\n"
            f"行业: {ctx.get('industry', '未知')}\n"
            f"分析区间: {ctx.get('start_date', '')} ~ {ctx.get('end_date', '')}\n"
        )
        similar_block = f"\n【历史相似场景参考】\n{similar}\n" if similar else ""
        factor_block = self._format_factor_block(ctx.get("factor_df"))
        signal_info = f"模型预测信号: {ctx.get('signal_value', 'N/A')}"

        if agent_type == "fundamental":
            prompt = (
                f"{stock_info}"
                f"财务因子数据:\n{factor_block}\n"
                f"{signal_info}\n"
                f"{similar_block}"
                "请从基本面角度判断该股票的短期方向。\n"
                f"{_JSON_SUFFIX}"
            )
        elif agent_type == "technical":
            prompt = (
                f"{stock_info}"
                f"技术指标数据:\n{factor_block}\n"
                f"{signal_info}\n"
                f"{similar_block}"
                "请从技术面角度判断该股票的短期方向。\n"
                f"{_JSON_SUFFIX}"
            )
        elif agent_type == "sentiment":
            prompt = (
                f"{stock_info}"
                f"风险指标:\n{factor_block}\n"
                f"{signal_info}\n"
                f"{similar_block}"
                "请从市场情绪角度判断当前是否适合参与该股票。\n"
                f"{_JSON_SUFFIX}"
            )
        else:  # news
            news_items = ctx.get("news_items", [])
            news_text = "\n".join(news_items[:10]) if news_items else "无近期新闻"
            prompt = (
                f"{stock_info}"
                f"近期新闻:\n{news_text}\n"
                f"{signal_info}\n"
                f"{similar_block}"
                "请从舆情角度判断该股票的短期方向。\n"
                f"{_JSON_SUFFIX}"
            )
        return prompt

    @staticmethod
    def _format_factor_block(factor_df) -> str:
        """将因子 DataFrame 格式化为文本块。"""
        if factor_df is None:
            return "无因子数据"
        try:
            import pandas as pd
            if isinstance(factor_df, pd.DataFrame) and not factor_df.empty:
                latest = factor_df.iloc[-1]
                lines = [f"  {k}: {v:.4f}" for k, v in latest.items() if isinstance(v, (int, float))]
                return "\n".join(lines[:15])
        except Exception:
            pass
        return str(factor_df)[:500]

    # ------------------------------------------------------------------
    # 辩论 + 风控 prompt
    # ------------------------------------------------------------------

    def debate(self, agent_outputs: dict[str, dict]) -> dict:
        """多空辩论 — 综合 4 个 Agent 的输出。"""
        system = "你是一位资深投资决策者，需要综合多位分析师的观点做出最终判断。"
        lines = []
        for name, out in agent_outputs.items():
            name_cn = {"fundamental": "基本面", "technical": "技术面",
                       "sentiment": "情绪面", "news": "舆情面"}.get(name, name)
            lines.append(
                f"【{name_cn}分析师】\n"
                f"信号: {out.get('signal','')}\n"
                f"置信度: {out.get('confidence',0)}\n"
                f"分析: {out.get('reasoning','')}\n"
            )
        user = (
            "以下是多位分析师对同一股票的观点，请综合判断:\n\n"
            + "\n".join(lines)
            + "\n请列出多方和空方主要论据，给出综合信号和置信度。\n"
            + _DEBATE_JSON_SUFFIX
        )
        raw = self.chat(system, user)
        return self._parse_json(raw)

    def risk_review(
        self,
        debate_result: dict,
        risk_metrics: dict,
        similar_cases: str = "",
    ) -> dict:
        """风控审查 — 审核最终投资建议。"""
        system = "你是一位严格的风控总监，负责审核投资建议的风险。"
        similar_block = f"\n【历史相似风险场景】\n{similar_cases}\n" if similar_cases else ""
        user = (
            "请审核以下投资建议的风险:\n\n"
            f"【建议】\n信号: {debate_result.get('signal','')}\n"
            f"置信度: {debate_result.get('confidence',0)}\n"
            f"综合判断: {debate_result.get('reasoning','')}\n\n"
            f"【风控指标】\n"
            f"VaR95: {risk_metrics.get('var_95','N/A')}\n"
            f"CVaR95: {risk_metrics.get('cvar_95','N/A')}\n"
            f"压力测试最大回撤: {risk_metrics.get('stress_max_dd','N/A')}\n"
            f"日亏损限额状态: {risk_metrics.get('daily_loss_ok','N/A')}\n"
            f"{similar_block}"
            f"请给出风险评级和 go/no-go 决定。\n"
            f"{_RISK_JSON_SUFFIX}"
        )
        raw = self.chat(system, user)
        return self._parse_json(raw)

    # ------------------------------------------------------------------
    # 最终报告生成
    # ------------------------------------------------------------------

    def generate_report(self, state: dict) -> str:
        """生成完整 Markdown 格式分析报告。"""
        system = "你是一位专业的金融报告撰写人，擅长将量化分析结果整理为结构清晰的 Markdown 报告。"
        user = self._build_report_prompt(state)
        return self.chat(system, user, temperature=0.5, max_tokens=4096)

    def _build_report_prompt(self, state: dict) -> str:
        """构建报告 prompt。"""
        return (
            f"# SmartAlpha Pro 智能分析报告\n\n"
            f"## 基本信息\n"
            f"- 股票: {state.get('stock_name','')} ({state.get('stock_code','')})\n"
            f"- 行业: {state.get('industry','')}\n"
            f"- 分析区间: {state.get('start_date','')} ~ {state.get('end_date','')}\n"
            f"- 分析深度: {state.get('analysis_depth','standard')}\n\n"
            f"## 模型信号\n{state.get('signal_value','N/A')}\n\n"
            f"## 多 Agent 分析\n"
            + self._format_agent_outputs(state.get("agent_outputs", {}))
            + f"\n## 综合判断\n"
            f"信号: {state.get('debate_result',{}).get('signal','')}\n"
            f"置信度: {state.get('debate_result',{}).get('confidence',0)}\n\n"
            f"## 风控审核\n"
            f"风险等级: {state.get('risk_assessment',{}).get('risk_level','')}\n"
            f"审核决定: {state.get('risk_assessment',{}).get('go_no_go','')}\n\n"
            "请将以上信息整理为一篇专业的 Markdown 格式投资分析报告，"
            "包含摘要、多维度分析、风险提示和操作建议四个章节。"
        )

    @staticmethod
    def _format_agent_outputs(agent_outputs: dict) -> str:
        """格式化 Agent 输出为文本。"""
        if not agent_outputs:
            return "暂无 Agent 分析结果\n"
        name_map = {"fundamental": "基本面", "technical": "技术面",
                     "sentiment": "情绪面", "news": "舆情面"}
        lines = []
        for name, out in agent_outputs.items():
            cn = name_map.get(name, name)
            lines.append(
                f"### {cn}分析\n"
                f"- 信号: {out.get('signal','')}\n"
                f"- 置信度: {out.get('confidence',0)}\n"
                f"- 分析: {out.get('reasoning','')}\n"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON 解析
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_agent_output(raw: str, agent_type: str = "") -> dict:
        """解析 Agent JSON 输出，失败时返回 safe default。"""
        result = DeepSeekClient._parse_json(raw)
        if not result or "signal" not in result:
            return {
                "signal": "neutral",
                "confidence": 0.5,
                "reasoning": f"{agent_type} Agent 解析失败，返回默认值",
                "key_metrics": {},
                "risk_flags": ["parse_error"],
            }
        return result

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """从 LLM 输出中提取 JSON。"""
        if not raw:
            return {}
        # 尝试提取 {...} 块
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass
        return {}
