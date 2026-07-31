# SmartAlpha Pro

A 股智能选股系统 — 用因子工程找信号，用 LLM Agent 做分析，用 A 股费率做回测。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-299%2F299-brightgreen)](test_results.txt)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 这个项目做什么

一句话：输入一只 A 股，系统自动完成**数据拉取 → 因子计算 → 模型预测 → 多 Agent 分析 → 回测评估 → 投资报告**的全流程。

拆开来看三层：

| 层 | 做什么 | 怎么做 |
|---|---|---|
| 因子层 | 把行情数据变成量化信号 | 自研表达式引擎，55 个金融函数，表达式编译执行 |
| 分析层 | 让 AI 从多角度评估股票 | 4 个 LLM Agent 并行分析，辩论合成结论 |
| 回测层 | 验证策略能不能赚钱 | A 股真实费率，7 项风控，VaR/压力测试 |

---

## 架构

```
                    ┌─────────────┐
                    │  Streamlit  │  ← 用户输入股票代码
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │    LangGraph 工作流       │
              │                        │
              │  Stage1: 数据+因子计算    │
              │     │                  │
              │     ├─ 基本面 Agent      │
              │     ├─ 技术面 Agent      │  ← 4 个 LLM Agent 并行
              │     ├─ 情绪面 Agent      │
              │     └─ 舆情面 Agent      │
              │     │                  │
              │  Stage3: 辩论+风控审核   │
              │     │                  │
              │  Stage4: 生成 Markdown  │
              │           报告+存储记忆  │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
   ┌─────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
   │ 因子引擎   │   │  ML 模型    │   │  回测引擎    │
   │ 表达式解析 │   │ LightGBM    │   │ A股费率/风控 │
   │ 55个函数   │   │ Transformer │   │ VaR/压力测试 │
   └───────────┘   └─────────────┘   └─────────────┘
```

---

## 核心能力

### 因子表达式引擎

不用写代码，用表达式定义因子：

```python
# 动量+波动率复合因子
"RANK(ROC(close, 20)) * 0.6 + RANK(-STD(close, 20)) * 0.4"
```

内部走完整编译链路：词法分析 → 语法解析（LL(1)递归下降）→ AST 执行，内置 55 个金融函数（统计、排名、时序、技术指标、截面），支持嵌套和条件计算。

### LLM 多 Agent 分析

接入 DeepSeek API，4 个专业 Agent 并行分析同一只股票，从基本面、技术面、情绪面、舆情面各自出报告，然后自动辩论合成一致结论，最后风控 Agent 审核，输出完整的 Markdown 投资报告。

- 基于 LangGraph TypedDict State，fan-out 并行零冲突
- ChromaDB 存储历史决策和反思，支持记忆检索
- Mock 模式：没 API Key 也能跑（用模拟数据演示工作流）

### Walk-Forward 模型训练

滚动窗口训练，严格防止未来信息泄漏：
- 每期用过去 N 天训练，预测下一期，窗口向前滚动
- Purge 间隔 5 天，避免标签重叠
- Optuna 超参搜索 + 时序交叉验证
- LightGBM + Transformer 双模型 Stacking 集成

### A 股回测

真实费率模型，不是拍脑袋的千分之一：

| 费用项 | 费率 |
|--------|------|
| 佣金 | 0.03%（万三） |
| 印花税 | 0.05%（卖出） |
| 滑点 | 0.1% |
| 冲击成本 | 成交额平方根模型 |

7 项风控规则：止损/止盈/仓位/行业集中度/黑名单/因子暴露。VaR/CVaR 支持历史模拟、参数法、蒙特卡洛三种算法。压力测试覆盖 2015 股灾、2020 疫情、2022 调整、2024 闪崩。

---

## 快速开始

```bash
git clone https://github.com/Weizikai-1/SmartAlpha-Pro
cd SmartAlpha-Pro
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env → 填入 Tushare Token + DeepSeek API Key

# 运行测试（无需 API Key）
python -m pytest tests/ -v --tb=short

# 启动界面
streamlit run smartalpha/ui/app.py
```

---

## 项目结构

```
smartalpha/
├── core/          # 因子表达式引擎（词法→语法→AST→执行）
├── graph/         # LangGraph 工作流 + TypedDict State
├── agents/        # 4 个专业分析 Agent
├── llm/           # DeepSeek API 封装（OpenAI 兼容）
├── memory/        # ChromaDB 向量记忆
├── model/         # LightGBM / Transformer / Optuna / Stacking
├── backtest/      # A 股回测引擎（费率+风控+VaR）
├── risk/          # 7 项风控规则 + 压力测试
├── factor/        # 因子工程（中性化/IC筛选/Mask）
├── data/          # Tushare + AKShare 数据获取
├── ui/            # Streamlit 交互界面
├── rl/            # SAC 强化学习（预留）
├── eval/          # 因子评估
├── registry/      # 因子注册中心
├── storage/       # 列式存储（LRU 缓存）
├── strategy/      # 多策略对比
└── pipeline.py    # 全流程管道
```

---

## 测试

```
tests/test_lexer.py            78 passed    词法分析
tests/test_executor.py         42 passed    AST 执行
tests/test_functions.py        68 passed    55 函数库
tests/test_factor.py           24 passed    因子工程
tests/test_model.py            11 passed    ML 模型
tests/test_backtest.py         23 passed    回测引擎
tests/test_cache.py            36 passed    LRU 缓存
tests/test_integration.py      17 passed    集成测试
─────────────────────────────────────────────────
                              299 passed ✓
```

---

## 技术栈

| 层 | 技术 |
|------|------|
| 数据 | Pandas, NumPy, PyArrow, Tushare Pro, AKShare |
| ML | LightGBM, Scikit-learn, Optuna |
| DL | PyTorch (Transformer) |
| LLM | DeepSeek API, LangGraph, ChromaDB |
| 前端 | Streamlit, Plotly |
| 部署 | Docker |

---

## License

MIT

---

**免责声明**: 本项目仅供学习和研究使用，不构成任何投资建议。
