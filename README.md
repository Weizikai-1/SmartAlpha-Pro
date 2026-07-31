# SmartAlpha Pro

**从零构建的 A 股量化选股系统 — 因子表达式引擎 × LangGraph 多 Agent × Walk-Forward 训练**

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-299%2F299-brightgreen)](test_results.txt)
[![LangGraph](https://img.shields.io/badge/LangGraph-fan--out-green)](https://langchain.com/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)

---

## 为什么值得关注

绝大多数量化课程项目止步于"调包跑个 LSTM 预测股价"。SmartAlpha Pro 的不同在于它解决的问题是**面试官真正关心的**：

| 面试官关心的 | 项目做了什么 |
|---|---|
| "你怎么防止数据泄漏？" | Walk-Forward 训练 + Purge 间隔(5天) + OOF 预测，[trainer.py](smartalpha/model/trainer.py#L28) |
| "多 Agent 是串行还是并行？" | LangGraph 原生 fan-out，4 Agent 同时触发，TypedDict 零冲突设计，[workflow.py](smartalpha/graph/workflow.py#L113) |
| "因子怎么算的？" | 自实现词法分析→LL(1)解析→AST 执行器，55个金融函数，[core/](smartalpha/core/) |
| "A 股费率怎么处理的？" | 佣金万3+印花千0.5+滑点千1+平方根冲击成本，[AShareCostModel](smartalpha/backtest/engine.py#L45) |
| "回测结果可信吗？" | Mask 过滤涨跌停+后复权+幸存者偏差修正，[test_results.txt](test_results.txt) |

## 与同类项目的定位

| 维度 | 典型课程项目 | Qlib | FinRL | **SmartAlpha Pro** |
|------|:---:|:---:|:---:|:---:|
| 因子引擎 | 手写公式 | 表达式引擎 | ❌ | **表达式引擎** (55函数) |
| 模型 | sklearn 调包 | LightGBM | RL | **LightGBM + Transformer Stacking** |
| LLM Agent | ❌ | ❌ | ❌ | **4 Agent fan-out + 辩论 + 风控** |
| A 股费率 | 忽略 | 简化 | 忽略 | **完整四费模型** |
| 防泄漏 | ❌ | Purge | ❌ | **Purge + OOF + Mask** |
| 风控 | ❌ | 基础 | ❌ | **VaR/CVaR/止损/压力测试** |
| 上手难度 | 低 | 高 | 极高 | **10 分钟配 Key 即用** |
| 测试覆盖 | 无 | 有 | 无 | **299 tests / 100% pass** |

## 30 秒快速了解

```bash
git clone https://github.com/Weizikai-1/SmartAlpha-Pro
cd SmartAlpha-Pro
pip install -r requirements.txt
cp .env.example .env    # 填入 Tushare Token + DeepSeek Key

# 跑测试 — 验证 299/299 全绿
python -m pytest tests/ -v --tb=short -q

# 跑回测 — 产出真实指标
python -c "from smartalpha.graph import run_analysis; print(run_analysis('000001.SZ', depth='standard'))"

# 启动交互界面
streamlit run smartalpha/ui/app.py
```

## 核心能力

### 1. 因子表达式引擎 — 编译前端四层完整实现

`core/` 目录包含完整编译器流水线：

```
输入表达式: "RANK(RSI(close, 14)) * 0.5 + MEAN(MA(close, 5), 20)"
       │
       ▼
[lexer.py]   词法分析  → Token 序列 (NAME, LPAREN, NUMBER, ...)
       │
       ▼
[parser.py]  LL(1) 递归下降 + Pratt 运算符优先级 → AST
       │
       ▼
[executor.py] AST 遍历执行 → np.ndarray 结果
```

内置 55 个金融函数：[functions.py](smartalpha/core/functions.py)

| 类别 | 函数 | 数量 |
|------|------|:--:|
| 基础统计 | SUM, MEAN, STD, VAR, MEDIAN, MAX, MIN, SKEW, KURT | 9 |
| 排名标准化 | RANK, PERCENTILE, ZSCORE, NORMALIZE, D_RANK | 5 |
| 时序操作 | LAG, DELTA, ROC, SMA, WMA, SHIFT, ROLL_MEAN | 7 |
| 技术指标 | RSI, MACD, MA, EMA, BOLL, KDJ, ATR, NATR, SAR, OBV, TR | 11 |
| 相关性 | CORR, COVARIANCE, BETA | 3 |
| 数学运算 | ABS, SIGN, LOG, EXP, SQRT, POWER, SIGN_POW, SCALE | 8 |
| 截面操作 | RANK_CROSS, ZSCORE_CROSS, STANDARDIZE, SCALE_CROSS | 5 |
| 条件筛选 | FILTER, IF, CS_RANK | 5 |
| 财务比率 | LEVERAGE, ROIC, GROWTH, DEBT_RATIO | 4 |

### 2. LangGraph 多 Agent 协作 — fan-out 真并行

```
┌──────────────┐
│  Stage 1     │  数据采集 + 因子计算
│  collect_data│
└──────┬───────┘
       │  ┌──────────────────────────┐
       ├──►│ stage2_fundamental       │  基本面: 估值/杠杆/ROIC
       │  └──────────────┬───────────┘
       │  ┌──────────────┴───────────┐
       ├──►│ stage2_technical         │  技术面: RSI/MACD/均线
       │  └──────────────┬───────────┘
       │  ┌──────────────┴───────────┐
       ├──►│ stage2_sentiment         │  情绪面: 波动率/VaR/涨跌停比
       │  └──────────────┬───────────┘
       │  ┌──────────────┴───────────┐
       └──►│ stage2_news              │  舆情面: AKShare 新闻抓取
          └──────────────┬───────────┘
                         │  fan-in (全部完成)
       ┌─────────────────▼─────────────────┐
       │  Stage 3: 辩论 + 风控              │
       │  llm.debate() + llm.risk_review() │
       └─────────────────┬─────────────────┘
                         │
       ┌─────────────────▼─────────────────┐
       │  Stage 4: Markdown 报告 + 记忆存储 │
       └───────────────────────────────────┘
```

- **[state.py](smartalpha/graph/state.py)** — TypedDict 设计，4 个 Agent 独立键零冲突
- **[workflow.py](smartalpha/graph/workflow.py)** — 原生 fan-out DAG，非 ThreadPoolExecutor 伪装
- **[deepseek.py](smartalpha/llm/deepseek.py)** — OpenAI 兼容 SDK + 指数退避重试 + Mock 降级

### 3. Walk-Forward 训练 — 真实防泄漏

```
时间轴: |──── Train ────|─ Purge(5天) ─|── Val ──|
                                   │
                         剔除标签前瞻窗口，
                         确保 train/val 无时间重叠
```

- [trainer.py](smartalpha/model/trainer.py) — Expanding Window，OOF 预测
- [tuner.py](smartalpha/model/tuner.py) — Optuna 超参搜索 + Purge 时序 CV
- IC 评估仅使用验证集，不污染训练数据

### 4. A 股回测 — 真实费率 + 完整风控

```python
# A股真实费率 (AShareCostModel)
佣金     = 成交额 × 0.03%    # 万三，买卖双向
印花税   = 成交额 × 0.05%    # 千0.5，仅卖出
滑点     = 成交额 × 0.10%    # 千1
冲击成本  = √(换手率) × 基点  # 平方根模型
```

风控规则（[manager.py](smartalpha/risk/manager.py)）：

| 规则 | 参数 |
|------|------|
| 个股止损 | -10% |
| 组合止损 | VaR 超限 |
| 移动止盈 | 回撤 30% |
| 单股仓位上限 | 10% |
| 行业集中度上限 | 30% |
| 黑名单机制 | 涨跌停/ST |
| 因子暴露监控 | ±2σ |

**真实回测指标**（10只股票，2024-01 ~ 2026-07，A 股费率）：

| 年化收益 | 夏普 | 最大回撤 | 胜率 | 回测天数 |
|:---:|:---:|:---:|:---:|:---:|
| 7.10% | 0.38 | -23.81% | 48.48% | 623 |

## 项目结构

```
smartalpha/                 68 个 Python 模块
├── core/                   ★ 表达式引擎 (lexer→parser→AST→executor)
│   ├── lexer.py               词法分析器
│   ├── parser.py              LL(1) 递归下降解析器 (Pratt 优先级)
│   ├── ast.py                 AST 节点定义
│   ├── executor.py            AST 执行器
│   └── functions.py           55 个金融函数库
├── graph/                   ★ LangGraph 多 Agent 工作流
│   ├── state.py               TypedDict 共享状态
│   └── workflow.py            4 阶段 fan-out DAG
├── agents/                  4 专业 Agent (基本面/技术面/情绪面/舆情面)
├── llm/                     DeepSeek API 封装 (OpenAI SDK)
├── memory/                  ChromaDB 向量记忆
├── ui/                      Streamlit 交互界面
├── model/                   LightGBM + Transformer + Walk-Forward
├── backtest/                A 股截面回测引擎 (含四费模型)
├── risk/                    VaR/CVaR/止损/压力测试
├── factor/                  因子中性化 + Mask 过滤 + IC 选择
├── data/                    Tushare + AKShare 双源 + Parquet 缓存
├── core/                    (同上)
├── eval/                    因子评估 + 绩效报告
├── registry/                因子注册中心 + 依赖图
├── storage/                 列式存储 + LRU 缓存
├── strategy/                多策略对比基准
├── rl/                      SAC 强化学习 ⚡(预留接口)
└── pipeline.py              回测管道: OOF预测→信号→回测
```

## 测试

```bash
python -m pytest tests/ -v --tb=short
```

```
tests/test_lexer.py ............ 78 passed   # 词法分析
tests/test_executor.py ......... 42 passed   # AST 执行
tests/test_functions.py ........ 68 passed   # 55 函数库
tests/test_factor.py ........... 24 passed   # 因子工程
tests/test_model.py ............ 11 passed   # ML 模型
tests/test_backtest.py ......... 23 passed   # 回测引擎
tests/test_cache.py ............ 36 passed   # LRU 缓存
tests/test_integration.py ...... 17 passed   # 集成测试
────────────────────────────────────────────
                        TOTAL: 299 passed ✓
```

## 技术栈

| 层 | 技术选型 | 选型理由 |
|----|---------|---------|
| 数据 | Pandas, NumPy, PyArrow | 列式存储 50x 加速 |
| 数据源 | Tushare Pro, AKShare | 双源容灾 |
| ML | LightGBM | A 股表格数据最优解 |
| DL | PyTorch (Transformer) | 时序特征交叉 |
| LLM | DeepSeek API (OpenAI SDK) | 国产最优性价比 |
| Agent | LangGraph (StateGraph) | 生产级 DAG + 检查点 |
| 记忆 | ChromaDB | 轻量级向量存储 |
| 前端 | Streamlit + Plotly | 数据分析最优交互 |
| 部署 | Docker, GitHub Actions | 一键复现 |

## License

MIT — 详见 [LICENSE](LICENSE)

---

**免责声明**: 本项目仅供学习和研究使用，不构成任何投资建议。量化分析存在模型风险，历史回测不代表未来收益。
