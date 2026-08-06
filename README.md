# SmartAlpha Pro

A 股多因子量化选股系统 — 因子工程、机器学习、LLM 多 Agent 分析、强化学习仓位管理的全链路投研平台。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-544%2F544-brightgreen)](https://github.com/Weizikai-1/SmartAlpha-Pro/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![LightGBM](https://img.shields.io/badge/ML-LightGBM-orange)](https://lightgbm.readthedocs.io/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-purple)](https://langchain-ai.github.io/langgraph/)

---

## 项目概述

输入一只 A 股代码，系统自动完成 **数据获取 → 因子计算 → 模型预测 → 多 Agent 分析 → 回测评估** 的全流程闭环。

| 层 | 职责 | 技术方案 |
|---|---|---|
| **数据层** | 行情/基本面/指数/行业多源数据获取与清洗 | Tushare Pro + AKShare, PyArrow 列式存储 |
| **因子层** | 92 个量价因子计算 + 表达式引擎 | 自研 LL(1) 编译器, 55 个金融函数, 行业市值中性化 |
| **模型层** | 截面收益预测 + 集成学习 | LightGBM Walk-Forward, Transformer, Optuna 超参搜索 |
| **分析层** | 多维度智能分析 + 投资报告 | LangGraph 4-Agent 并行, DeepSeek LLM, ChromaDB 记忆 |
| **执行层** | A 股真实费率回测 + 风控 + RL 仓位管理 | 7 项风控规则, VaR/CVaR, SAC 强化学习 |
| **展示层** | Web 交互界面 | Streamlit + Plotly |

---

## 架构

```
                         ┌──────────────────┐
                         │  Streamlit 前端    │  ← 用户交互
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │     LangGraph 工作流         │
                    │                            │
                    │  基本面 Agent ─┐           │
                    │  技术面 Agent ─┤           │
                    │  情绪面 Agent ─┼─ 辩论 → 风控 │  ← 4 Agent 并行
                    │  舆情面 Agent ─┘           │
                    │                            │
                    │  ChromaDB 记忆存储          │
                    └─────────────┬──────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
  ┌─────▼──────┐          ┌──────▼──────┐          ┌───────▼──────┐
  │ 因子引擎     │          │  ML 模型     │          │  回测引擎      │
  │ 表达式编译   │          │ LightGBM    │          │  A股费率模型   │
  │ 55 金融函数  │          │ Transformer │          │  7 项风控      │
  │ 中性化/Mask │          │ Optuna/SAC  │          │  VaR/压力测试  │
  └────────────┘          └─────────────┘          └──────────────┘
```

---

## 核心能力

### 因子表达式引擎

自定义 DSL 定义复合因子，表达式实时编译执行：

```python
# 动量 + 低波复合因子
"RANK(ROC(close, 20)) * 0.6 + RANK(-STD(close, 20)) * 0.4"
```

词法分析 → LL(1) 语法解析 → AST 执行，内置 55 个金融函数，支持嵌套、条件计算、截面/时序操作。

### LLM 多 Agent 智能分析

基于 LangGraph 的多 Agent 协作系统：
- 4 个专业 Agent（基本面/技术面/情绪面/舆情面）并行分析
- 自动辩论合成一致结论
- 风控 Agent 终审
- ChromaDB 持久化记忆，支持历史决策回溯
- 输出结构化 Markdown 投资报告
- **Mock 模式**：无 API Key 也能演示完整工作流

### Walk-Forward 模型训练

严格时序分割，防止未来信息泄漏：
- 滚动窗口训练：每期 N 天训练 → 下期预测
- Purge 间隔 5 天，避免标签重叠
- LightGBM + Transformer Stacking 集成
- Optuna 超参数自动搜索

### A 股真实费率回测

| 费用项 | 费率 |
|--------|------|
| 佣金 | 0.03%（万三） |
| 印花税 | 0.05%（卖出） |
| 滑点 | 0.1% |

7 项风控：止损/止盈/仓位限制/行业集中度/黑名单/因子暴露/连续亏损熔断。
VaR/CVaR 支持历史模拟法、参数法、蒙特卡洛三种算法。
压力测试覆盖 2015 股灾、2020 疫情、2022 调整等极端场景。

### 强化学习仓位管理

基于 SAC（Soft Actor-Critic）的动态仓位优化：
- PortfolioEnv：11 维状态空间（7 因子 + 3 市场 + 1 持仓）
- 连续动作空间 [-1, 1] 映射仓位调整比例
- 奖励函数：收益 + 回撤惩罚

---

## 回测结果

基于 HS300 成分股（41 只），2024.01 — 2026.08，月度调仓 Top 10：

| 指标 | 无风控 | 含风控 |
|------|--------|--------|
| 年化收益率 | 8.95% | 1.21% |
| 夏普比率 | 0.67 | 0.67 |
| 最大回撤 | -17.34% | -5.03% |
| Calmar 比率 | 0.52 | 0.24 |
| 风控触发 | — | 256 次 |

> 风控模式下回撤大幅收窄（-17.34% → -5.03%），验证了风控系统的有效性。

---

## 快速开始

```bash
git clone https://github.com/Weizikai-1/SmartAlpha-Pro.git
cd SmartAlpha-Pro
pip install -e .

# 配置 API Key（可选，Mock 模式无需）
cp .env.example .env

# 运行测试
pytest tests/ -q --ignore=tests/test_memory.py

# 启动 Web 界面
streamlit run smartalpha/ui/app.py
```

---

## 项目结构

```
SmartAlpha-Pro/
├── smartalpha/
│   ├── config.py         # 统一配置管理（路径/常量/API密钥）
│   ├── core/             # 因子表达式引擎（词法→语法→AST→执行）
│   ├── factor/           # 因子工程（中性化/Mask/IC筛选/相关性去重）
│   ├── model/            # ML 模型（LightGBM/Transformer/Optuna/Stacking）
│   ├── rl/               # SAC 强化学习仓位管理
│   ├── backtest/         # A 股回测引擎（费率+风控+VaR）
│   ├── risk/             # 7 项风控规则 + 压力测试
│   ├── graph/            # LangGraph 工作流 + State 管理
│   ├── agents/           # 4 个专业分析 Agent + 辩论
│   ├── llm/              # DeepSeek API 封装
│   ├── memory/           # ChromaDB 向量记忆
│   ├── data/             # 多源数据获取（Tushare/AKShare）
│   ├── ui/               # Streamlit Web 界面
│   ├── registry/         # 因子注册中心
│   ├── storage/          # 列式存储 + LRU 缓存
│   └── pipeline.py       # 全流程管道
├── scripts/              # 工具脚本
│   ├── download_data.py  # 数据下载
│   ├── rl_train.py       # RL 训练
│   └── e2e_test.py       # 端到端验证
├── tests/                # 30 个测试文件，544 个测试用例
├── pyproject.toml        # 项目配置
├── requirements.txt      # 依赖清单
├── Dockerfile            # Docker 部署
└── .github/workflows/    # CI 流水线
```

---

## 测试

| 模块 | 用例 | 覆盖范围 |
|------|------|---------|
| 词法分析器 | 78 | Token 解析全覆盖 |
| AST 执行器 | 42 | 表达式执行 + 边界 |
| 函数库 | 68 | 55 个金融函数 |
| 因子工程 | 24 | 中性化/IC/相关性 |
| ML 模型 | 11 | LightGBM + WalkForward |
| 回测引擎 | 23 | 费率/风控/净值 |
| 缓存/存储 | 36 | LRU + 列式存储 |
| 集成测试 | 17 | 跨模块协作 |
| Mock 训练 | 15 | 模拟数据全链路 |
| **合计** | **544** | — |

```bash
pytest tests/ -q               # 完整测试
pytest tests/ -q --tb=short    # 简洁输出
```

---

## 技术栈

| 类别 | 技术 |
|------|------|
| 数据处理 | Pandas, NumPy, PyArrow (Parquet) |
| 数据源 | Tushare Pro, AKShare |
| 机器学习 | LightGBM, Scikit-learn, Optuna |
| 深度学习 | PyTorch (Transformer) |
| 强化学习 | Stable-Baselines3 (SAC) |
| LLM/Agent | LangGraph, LangChain, DeepSeek API, ChromaDB |
| Web 界面 | Streamlit, Plotly |
| 工程化 | pytest, Config 统一管理, CI (GitHub Actions) |
| 部署 | Docker |

---

## License

MIT

---

**免责声明**: 本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
