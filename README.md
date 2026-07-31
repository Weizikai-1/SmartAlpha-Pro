# SmartAlpha Pro

基于量化因子工程与大语言模型多智能体协作的 A 股智能选股系统。融合传统多因子模型与 LLM 驱动的 Agent 分析，覆盖数据采集、因子计算、模型训练、回测风控到投资报告生成的全链路。

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-299%2F299-brightgreen)](test_results.txt)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 项目概述

SmartAlpha Pro 做了三件事：

**1. 因子工程** — 自实现完整的因子表达式引擎（词法分析 → 语法解析 → AST 执行），内置 55 个金融函数，支持嵌套表达式和条件计算。配合 Walk-Forward 滚动训练、Purge 防泄漏、因子 IC 筛选，构建可验证的因子体系。

**2. 多 Agent 分析** — 基于 LangGraph 搭建 4 个专业 Agent（基本面 / 技术面 / 情绪面 / 舆情面）并行分析，辩论合成投资建议，风控审核后输出 Markdown 格式投资报告。

**3. A 股回测** — 完整的 A 股真实费率模型（佣金万三 + 印花税千 0.5 + 滑点千 1 + 冲击成本）、7 项风控规则（止损/止盈/仓位/行业集中度/黑名单）、VaR/CVaR 三种计算方法、4 种历史极端场景压力测试。

## 核心模块

### 因子表达式引擎

`smartalpha/core/` — 编译前端四层完整实现：

```
表达式: "RANK(RSI(close, 14)) * 0.5 + MEAN(MA(close, 5), 20)"

[lexer.py]    → Token 序列
[parser.py]   → 抽象语法树 (LL(1)递归下降 + Pratt 运算符优先级)
[executor.py] → 遍历 AST，执行计算
[functions.py]→ 55 个金融函数 (统计/排名/时序/技术指标/截面)
```

### LangGraph 多 Agent 工作流

`smartalpha/graph/` — 4 阶段 fan-out DAG：

```
Stage 1: 数据采集 + 因子计算
   │
   ├── stage2_fundamental    基本面分析 (估值/杠杆/ROIC)
   ├── stage2_technical      技术面分析 (RSI/MACD/均线/布林带)
   ├── stage2_sentiment      情绪面分析 (波动率/VaR/涨跌停比)
   └── stage2_news          舆情面分析 (新闻抓取+情绪提取)
   │
Stage 3: Agent 辩论 + 风控审核
   │
Stage 4: 生成 Markdown 投资报告 + ChromaDB 记忆存储
```

- State 使用 TypedDict 设计，各 Agent 写入独立 key，fan-out 并行零冲突
- LLM 通过 DeepSeek API (OpenAI 兼容 SDK) 调用，支持 Mock 降级模式
- 报告含摘要、多维度分析、风险提示、操作建议四个章节

### 模型训练

`smartalpha/model/` — 机器学习预测管线：

- **LightGBM** — 表格数据最优解，特征重要性评估
- **Transformer** (PyTorch) — 时序特征交叉，因果 Mask 防未来信息泄漏
- **Walk-Forward** — Expanding Window 滚动训练，Purge 间隔(5天)防止标签泄漏
- **Optuna 超参搜索** — 带 Purge 的时序交叉验证
- **Stacking 集成** — LightGBM + Transformer 加权融合

### 风控体系

`smartalpha/risk/` — 7 项风控规则 + VaR/CVaR：

| 规则 | 机制 |
|------|------|
| 个股止损 | 单日亏损 > 10% 触发 |
| 组合止损 | 组合日内亏损超 VaR 阈值 |
| 移动止盈 | 从峰值回撤 > 30% 止盈 |
| 仓位限制 | 单股 ≤ 10%，总仓位动态调整 |
| 行业集中度 | 单行业 ≤ 30% |
| 黑名单 | 涨跌停 / ST 股自动排除 |
| 因子暴露 | 组合暴露偏离 ±2σ 告警 |

VaR / CVaR 支持三种计算方法：历史模拟法、参数法、蒙特卡洛模拟。压力测试覆盖 2015 股灾、2020 疫情、2022 调整、2024 闪崩四种历史极端场景。

## 项目结构

```
smartalpha/                    # 68 个 Python 模块
├── core/                     # 因子表达式引擎
│   ├── lexer.py              #   词法分析器
│   ├── parser.py             #   LL(1) 递归下降解析器
│   ├── ast.py                #   AST 节点定义
│   ├── executor.py           #   执行器
│   ├── functions.py          #   55 金融函数库
│   └── _func_helpers.py      #   内部辅助函数
├── graph/                    # LangGraph 工作流
│   ├── state.py              #   TypedDict 共享状态
│   └── workflow.py           #   4 阶段 fan-out DAG
├── agents/                   # 专业分析 Agent
├── llm/                      # DeepSeek API 封装
├── memory/                   # ChromaDB 向量记忆
├── ui/                       # Streamlit 交互界面
├── model/                    # ML 模型层
├── backtest/                 # A 股回测引擎
├── risk/                     # 风控模块
├── factor/                   # 因子工程 (中性化/Mask/选择)
├── data/                     # 数据获取 (Tushare/AKShare)
├── eval/                     # 因子评估
├── registry/                 # 因子注册中心
├── storage/                  # 列式存储 (LRU 缓存)
├── strategy/                 # 多策略对比
├── rl/                       # SAC 强化学习 (预留)
└── pipeline.py               # 回测管道
```

## 快速开始

```bash
# 克隆项目
git clone https://github.com/Weizikai-1/SmartAlpha-Pro
cd SmartAlpha-Pro
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 Tushare Token 和 DeepSeek API Key
# Token 获取: https://tushare.pro  |  API Key: https://platform.deepseek.com

# 下载数据 (可选，已有 10 只股票 demo 缓存)
python scripts/download_data.py --start 20240101

# 运行测试
python -m pytest tests/ -v --tb=short

# 启动 Streamlit 界面
streamlit run smartalpha/ui/app.py

# 命令行分析
python -c "from smartalpha.graph import run_analysis; run_analysis('000001.SZ')"
```

## 测试

299 个测试用例，100% 通过率：

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

## 技术栈

| 层 | 技术 |
|------|------|
| 数据 | Pandas, NumPy, PyArrow, Tushare Pro, AKShare |
| ML | LightGBM, Scikit-learn, Optuna |
| DL | PyTorch (Transformer) |
| LLM | DeepSeek API, LangGraph, ChromaDB |
| 前端 | Streamlit, Plotly |
| 部署 | Docker, GitHub Actions |

## License

MIT

---

**免责声明**: 本项目仅供学习和研究使用，不构成任何投资建议。
