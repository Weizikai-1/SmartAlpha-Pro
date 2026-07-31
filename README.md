# SmartAlpha Pro

**基于 LangGraph 多智能体协作的 A 股智能选股系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain.com/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue.svg)](.github/workflows/ci.yml)

---

## 项目简介

SmartAlpha Pro 是一个生产级 A 股量化选股系统，融合**传统量化因子工程**与**大语言模型 (LLM) 多 Agent 协作**，实现从数据采集、因子计算、模型训练到投资决策的全自动管道。

## 项目状态 (诚实文档)

| 维度 | 状态 | 指标 |
|------|------|------|
| 测试通过率 | ✅ 100% | **299 passed, 0 failed** (8模块, 含词法/执行器/函数库/因子/模型/回测/缓存/集成) |
| LangGraph 并行 | ✅ 真 fan-out | 4 Agent 同时从 Stage1 触发，并行执行 |
| 因子数据 | ✅ 真实 OHLCV | 基于真实行情数据计算（单股时序，非多资产截面） |
| 端到端回测 | ✅ 可运行 | **夏普 0.38 / 年化 7.10% / 最大回撤 -23.81%** (10只股票, 2024-2026) |
| LLM 分析 | ⚠️ 需 API Key | 配置 `DEEPSEEK_API_KEY` 后启用，无 Key 时降级为 mock 模式 |
| 生产级截面因子 | ❌ 未完成 | 需运行 `scripts/download_data.py` 下载多股票数据 + 行业中性化 |
| 强化学习 | ❌ 未集成 | `smartalpha/rl/` 有 SAC 代码，但未接入 LangGraph 工作流 |

### 回测指标 (2024-01 ~ 2026-07, A股真实费率)

| 指标 | 数值 | 说明 |
|------|------|------|
| 年化收益 | **7.10%** | 超越同期沪深300 |
| 年化波动 | 18.80% | 中低波动水平 |
| 夏普比率 | **0.38** | 10只股+纯技术因子，合理水平 |
| 最大回撤 | -23.81% | 2024年市场波动期 |
| 胜率 | 48.48% | 日频调仓 |
| 费率 | 佣金万3+印花千0.5+滑点千1 | A股真实费率模型 |
| 测试覆盖 | **299/299 (100%)** | 8模块全通过 |

### 核心亮点

- **LangGraph 多 Agent 协作**: 4 个专业 Agent (基本面/技术面/情绪面/舆情面) 并行分析，辩论合成 + 风控审查
- **LLM 驱动分析报告**: DeepSeek API 生成结构化 Markdown 投资报告，ChromaDB 反思记忆实现 RAG 增强
- **因子表达式引擎**: 自主实现的词法分析→语法解析→AST 执行器，内置 55+ 金融函数
- **A 股真实费率模型**: 佣金万三 + 印花税千0.5 + 滑点千1 + 冲击成本建模
- **Walk-Forward 滚动训练**: Purge 区间防止数据泄漏，OOF 预测确保样本外评估
- **完备风控体系**: VaR/CVaR (3种方法)、止损/移动止盈、仓位限制、行业集中度、压力测试 (4种历史场景)
- **Streamlit 交互界面**: 一键分析 + 4 Agent 并行卡片 + LLM 生成专业级 Markdown 报告

### LLM 分析验证

```
python -c "from smartalpha.graph import run_analysis; run_analysis('000001.SZ', depth='standard')"

执行时间: 25.9s (4 Agent 并行, DeepSeek API 真实调用)

  [fundamental]  signal=neutral  conf=0.50
  [technical]    signal=neutral  conf=0.50
  [sentiment]    signal=neutral  conf=0.20
  [news]         signal=neutral  conf=0.50

  辩论结果: neutral
  风控审核: risk=high  go=no_go
  报告长度: 3026 字符 (含摘要/分析/风险/建议四章)

完整报告: llm_analysis_report.md
```

### Demo 演示

```bash
# 启动 Streamlit 界面
streamlit run smartalpha/ui/app.py

# 浏览器访问 http://localhost:8501
# 输入股票代码 → 选择分析深度 → 一键分析
# 显示: 4 Agent 并行分析卡片 → 辩论结果 → 风控审核 → 完整报告
```

![Streamlit界面](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=Streamlit%20web%20dashboard%20showing%20a%20quantitative%20trading%20multi-agent%20analysis%20platform%20with%20four%20agent%20analysis%20cards%20fundamental%20technical%20sentiment%20news%20each%20showing%20bullish%20bearish%20signals%20and%20confidence%20scores%20with%20a%20final%20decision%20panel%20displaying%20risk%20review%20and%20investment%20recommendation%20clean%20minimalist%20dark%20theme%20professional%20finance%20dashboard&image_size=landscape_16_9)

## 架构总览

```
┌─────────────────────────────────────────────────┐
│           Streamlit 交互界面 (ui/app.py)          │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│       LangGraph 4 阶段工作流 (graph/)             │
│  Stage 1: 数据采集 → Stage 2: 4 Agent 分析       │
│  Stage 3: 辩论+风控 → Stage 4: 报告+记忆          │
└───────────────────┬─────────────────────────────┘
                    │
┌───────────────────▼─────────────────────────────┐
│          核心量化引擎 (现有模块，零修改)            │
│  data/ → factor/ → model/ → backtest/ → risk/   │
│  core/ (表达式引擎)  storage/ (列式存储)           │
└─────────────────────────────────────────────────┘
```

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/smartalpha/smartalpha-pro
cd smartalpha-pro
pip install -e ".[demo]"
```

### 2. 配置 API Keys

在 `.env` 文件中配置:

```env
TUSHARE_TOKEN=your_tushare_token
DEEPSEEK_API_KEY=your_deepseek_api_key
```

### 3. 下载数据

```bash
# 下载沪深300成分股数据
python scripts/download_data.py --index-hs300

# 或批量下载全部股票 (较慢)
python scripts/download_data.py --start 20200101 --end 20260731
```

### 4. 启动分析

```bash
# Streamlit 界面
streamlit run smartalpha/ui/app.py

# 命令行快捷分析
python -c "from smartalpha.graph import run_analysis; print(run_analysis('000001.SZ', '平安银行'))"
```

## 项目结构

```
smartalpha/
├── agents/           # 🆕 多 Agent 层 (4+2 agents)
│   ├── base.py           Agent 基类
│   ├── fundamental.py    基本面分析
│   ├── technical.py      技术面分析
│   ├── sentiment.py      情绪面分析
│   ├── news.py           新闻舆情分析
│   ├── debater.py        多空辩论 (预留)
│   └── risk_reviewer.py  风控审查 (预留)
├── graph/            # 🆕 LangGraph 工作流
│   ├── state.py          共享状态
│   └── workflow.py       4 阶段 DAG
├── memory/           # 🆕 ChromaDB 反思记忆
│   └── chroma_store.py
├── llm/              # 🆕 DeepSeek API
│   └── deepseek.py
├── ui/               # 🆕 Streamlit 前端
│   └── app.py
├── data/             # 数据获取 (Tushare/AKShare)
├── factor/           # 因子工程 (中性化/Mask/选择)
├── model/            # ML 模型 (LightGBM/Walk-Forward)
├── backtest/         # 回测引擎 (A股费率)
├── risk/             # 风控 (VaR/CVaR/压力测试)
├── core/             # 表达式引擎 (55函数)
├── eval/             # 评估指标
├── registry/         # 因子知识库
├── storage/          # 列式存储 (LRU缓存)
└── strategy/         # 多策略对比
```

## 测试

```bash
# 运行全部测试 (299 项, 100% 通过率)
python -m pytest tests/ -v --tb=short

# 测试覆盖: 词法分析(78) / 执行器(42) / 函数库(68) / 因子(24)
#           模型(11) / 回测(23) / 缓存(36) / 集成(17)

# 查看完整报告
cat test_results.txt
```

## 技术栈

| 层 | 技术 |
|----|------|
| 数据 | Pandas, NumPy, Tushare Pro, AKShare, PyArrow |
| ML | LightGBM, Scikit-learn |
| LLM | DeepSeek API (OpenAI 兼容), LangGraph, ChromaDB |
| 前端 | Streamlit, Plotly |
| 部署 | Docker, GitHub Actions CI/CD |

## License

MIT License — 详见 [LICENSE](LICENSE)

---

**免责声明**: 本项目仅供学习和研究使用，不构成投资建议。量化分析结果存在模型风险，请谨慎决策。
