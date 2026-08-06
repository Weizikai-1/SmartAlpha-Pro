"""SmartAlpha Pro — 统一配置管理。

集中管理所有路径、常量、阈值和环境变量读取。
各模块通过 `from smartalpha.config import ...` 引用，不再硬编码。

设计原则:
- 只读: 所有属性通过 @property 或模块级常量暴露
- 懒加载: 环境变量首次访问时读取
- 向后兼容: 提供与原硬编码路径相同的 Path 对象
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 加载 .env（幂等，重复调用不重复读取）
load_dotenv()

# ═══════════════════════════════════════════════════════════════════
# 项目路径
# ═══════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
MODEL_SAVE_DIR = Path(__file__).parent / "model" / "saved"

# 确保目录存在（首次引用时创建）
for _d in (CACHE_DIR, MODEL_SAVE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ChromaDB 持久化目录
CHROMA_PERSIST_DIR = os.path.join(
    os.path.expanduser("~"), ".smartalpha", "chroma"
)

# ═══════════════════════════════════════════════════════════════════
# 交易日常数
# ═══════════════════════════════════════════════════════════════════

TRADING_DAYS_PER_YEAR = 252              # A 股年均交易日
EPS = 1e-10                               # 数值精度（避免除零）
LIMIT_THRESHOLD = 0.095                   # 涨跌停阈值（±10% 留 0.5% 容差）
MAX_MISSING_RATIO = 0.1                   # 数据缺失率上限
MIN_DATA_POINTS = 20                      # 最小有效数据点

# ═══════════════════════════════════════════════════════════════════
# API 密钥 & 外部服务
# ═══════════════════════════════════════════════════════════════════

TUSHARE_TOKEN = os.getenv("TUSHARE_TOKEN", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DATA_SOURCE = os.getenv("DATA_SOURCE", "tushare")  # tushare / akshare

# Redis（可选）
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# ═══════════════════════════════════════════════════════════════════
# DeepSeek LLM 默认参数
# ═══════════════════════════════════════════════════════════════════

LLM_MODEL = "deepseek-chat"
LLM_TIMEOUT = 60.0                        # 请求超时（秒）
LLM_MAX_RETRIES = 3                       # 指数退避最大重试次数
LLM_TEMPERATURE = 0.3                     # 分析类任务的推荐温度
LLM_MAX_TOKENS = 2048                     # 默认最大输出 token

# ═══════════════════════════════════════════════════════════════════
# 回测默认参数
# ═══════════════════════════════════════════════════════════════════

BACKTEST_START_DATE = os.getenv("BACKTEST_START_DATE", "20200101")
BACKTEST_END_DATE = os.getenv("BACKTEST_END_DATE", "20261231")
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1000000"))
COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.0003"))       # 万三
SLIPPAGE_RATE = float(os.getenv("SLIPPAGE_RATE", "0.001"))           # 千一
STAMP_DUTY_RATE = float(os.getenv("STAMP_DUTY_RATE", "0.001"))       # 千一（卖出）

# ═══════════════════════════════════════════════════════════════════
# 风控默认参数
# ═══════════════════════════════════════════════════════════════════

MAX_SINGLE_POSITION = float(os.getenv("MAX_POSITION_PCT", "0.1"))    # 单股 ≤ 10%
MAX_INDUSTRY_PCT = float(os.getenv("MAX_INDUSTRY_PCT", "0.3"))       # 单行业 ≤ 30%
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.1"))             # 个股止损 -10%
STOP_PROFIT_PCT = float(os.getenv("STOP_PROFIT_PCT", "0.3"))         # 止盈 +30%
VAR_CONFIDENCE = float(os.getenv("VAR_CONFIDENCE", "0.95"))          # VaR 置信度
DAILY_LOSS_LIMIT = -0.05                                              # 日亏损限额
MAX_MONTHLY_LOSS = -0.15                                              # 月度亏损限额
CONSECUTIVE_LOSS_DAYS = 3                                             # 连续亏损天数阈值
BLACKLIST_DAYS = 5                                                    # 止损后黑名单天数

# ═══════════════════════════════════════════════════════════════════
# 模型训练参数
# ═══════════════════════════════════════════════════════════════════

PURGE_DAYS = 5                              # Walk-Forward purge 间隔
VAL_DAYS = 60                               # 验证集天数
STEP_DAYS = 20                              # Walk-Forward 步长
MIN_TRAIN_DAYS = 120                        # 最小训练天数
EARLY_STOPPING_ROUNDS = 50                  # LightGBM early stopping
LGBM_N_ESTIMATORS = 200                     # LightGBM 默认迭代数
LGBM_LEARNING_RATE = 0.05                   # LightGBM 学习率
LGBM_NUM_LEAVES = 31                        # LightGBM 叶子数

# ═══════════════════════════════════════════════════════════════════
# RL 训练参数
# ═══════════════════════════════════════════════════════════════════

RL_TOTAL_TIMESTEPS = 10000                  # SAC 训练总步数
RL_LEARNING_RATE = 0.0003                   # SAC 学习率
RL_BUFFER_SIZE = 10000                      # Replay Buffer 大小
RL_BATCH_SIZE = 64                          # 批次大小

# ═══════════════════════════════════════════════════════════════════
# 表达式引擎
# ═══════════════════════════════════════════════════════════════════

MAX_EXPRESSION_DEPTH = 50                   # 表达式嵌套深度上限
MAX_FUNCTION_ARGS = 10                      # 函数参数数量上限

# ═══════════════════════════════════════════════════════════════════
# 日志 & 调试
# ═══════════════════════════════════════════════════════════════════

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENV = os.getenv("ENV", "development")       # development / production
