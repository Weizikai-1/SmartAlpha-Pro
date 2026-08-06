"""强化学习环境 - FinRL风格，仓位管理"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smartalpha.config import DATA_DIR, PROCESSED_DIR

logger = logging.getLogger(__name__)

# 兼容处理：尝试导入gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    logger.warning("gymnasium未安装，RL功能不可用。运行: pip install gymnasium stable-baselines3")
    HAS_GYM = False
    # 占位符
    class gym:
        class Env:
            pass
    class spaces:
        class Box:
            def __init__(self, *args, **kwargs):
                pass


def _safe_read_parquet(path):
    """兼容 pyarrow 版本差异的安全 parquet 读取"""
    try:
        return pd.read_parquet(path)
    except OSError:
        logger.debug("pyarrow 读取失败，尝试 fastparquet 引擎")
        return pd.read_parquet(path, engine="fastparquet")


class PortfolioEnv(gym.Env if HAS_GYM else object):
    """投资组合管理环境

    State: [因子值(7维), 市场状态(3维), 当前持仓(1维)]
    Action: 连续动作 [-1, 1] -> 映射为仓位调整比例
    Reward: 夏普比率 + 回撤惩罚
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, factor_file="factors_neutral.parquet",
                 pred_file="model_predictions.parquet",
                 initial_capital=1_000_000, n_stocks=10,
                 rebalance_days=5, transaction_cost=0.001):
        if HAS_GYM:
            super().__init__()
        self.initial_capital = initial_capital
        self.n_stocks = n_stocks
        self.rebalance_days = rebalance_days
        self.transaction_cost = transaction_cost

        # 加载数据
        self.daily = _safe_read_parquet(PROCESSED_DIR / "daily_masked.parquet")
        self.daily["trade_date"] = pd.to_datetime(self.daily["trade_date"], format="%Y%m%d")
        self.daily = self.daily.sort_values(["ts_code", "trade_date"])
        self.daily["ret"] = self.daily.groupby("ts_code")["close"].pct_change()

        try:
            self.factors = _safe_read_parquet(PROCESSED_DIR / factor_file)
            self.factors["trade_date"] = pd.to_datetime(self.factors["trade_date"], format="%Y%m%d")
        except FileNotFoundError:
            logger.error("缺少因子数据")
            raise

        try:
            self.preds = _safe_read_parquet(PROCESSED_DIR / pred_file)
            self.preds["trade_date"] = pd.to_datetime(self.preds["trade_date"], format="%Y%m%d")
        except FileNotFoundError:
            self.preds = None

        self.trade_dates = sorted(self.daily["trade_date"].unique())
        self.stock_list = sorted(self.daily["ts_code"].unique())
        self.n_stocks_total = len(self.stock_list)

        # 特征维度: 7因子 + 3市场状态 + 1持仓 = 11
        self.n_features = 11
        if HAS_GYM:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.n_features,), dtype=np.float32)
            self.action_space = spaces.Box(
                low=-1, high=1, shape=(1,), dtype=np.float32)
        else:
            self.observation_space = None
            self.action_space = None

        self.current_step = 0
        self.portfolio_value = initial_capital
        self.peak_value = initial_capital
        self.holdings = {}
        self.returns_history = []

    def reset(self, seed=None, options=None):
        if HAS_GYM:
            super().reset(seed=seed)
        self.current_step = 60
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        self.holdings = {}
        self.returns_history = []
        return self._get_obs(), {}

    def step(self, action):
        """执行动作并返回新状态"""
        action_val = float(np.clip(action[0], -1, 1))
        current_date = self.trade_dates[self.current_step]
        next_date_idx = min(self.current_step + self.rebalance_days, len(self.trade_dates) - 1)
        next_date = self.trade_dates[next_date_idx]

        # 根据动作选择股票数量
        n_select = max(1, min(self.n_stocks, int(self.n_stocks * (0.5 + 0.5 * action_val))))

        # 选股：模型预测分最高
        day_preds = self.preds[self.preds["trade_date"] == current_date] if self.preds is not None else None
        if day_preds is not None and not day_preds.empty:
            selected = day_preds.nlargest(n_select, "predict_score")["ts_code"].tolist()
        else:
            rng = np.random.RandomState(self.current_step)
            selected = rng.choice(self.stock_list, n_select, replace=False).tolist()

        # 计算持仓期收益
        period_data = self.daily[
            (self.daily["trade_date"] > current_date) &
            (self.daily["trade_date"] <= next_date) &
            (self.daily["ts_code"].isin(selected))
        ]

        if period_data.empty:
            daily_ret = 0.0
        else:
            daily_rets = period_data.groupby("trade_date")["ret"].mean().dropna()
            if daily_rets.empty:
                daily_ret = 0.0
            else:
                # 计算实际换手率
                prev_holdings = set(self.holdings.keys()) if self.holdings else set()
                new_holdings = set(selected)
                if prev_holdings:
                    turnover = len(new_holdings.symmetric_difference(prev_holdings)) / max(len(new_holdings), 1)
                else:
                    turnover = 1.0
                cost = turnover * self.transaction_cost / self.rebalance_days
                daily_rets = daily_rets - cost
                # 正确：用累计收益更新净值
                portfolio_ret = (1 + daily_rets).prod() - 1
                daily_ret = portfolio_ret

        # 更新组合价值
        self.portfolio_value *= (1 + daily_ret)
        self.holdings = {s: 1.0 / max(len(selected), 1) for s in selected}
        self.peak_value = max(self.peak_value, self.portfolio_value)
        self.returns_history.append(daily_ret)

        # 计算奖励: 日收益 - 回撤惩罚
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
        reward = daily_ret * 100 - drawdown * 50

        # 前进
        self.current_step += self.rebalance_days
        terminated = False
        truncated = self.current_step >= len(self.trade_dates) - self.rebalance_days

        obs = self._get_obs()
        info = {
            "portfolio_value": self.portfolio_value,
            "daily_return": daily_ret,
            "drawdown": drawdown,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """构建状态向量"""
        current_date = self.trade_dates[self.current_step]

        # 因子特征 (取截面均值)
        day_factors = self.factors[self.factors["trade_date"] == current_date]
        factor_cols = [c for c in day_factors.columns
                       if c not in ["ts_code", "trade_date", "industry", "circ_mv", "log_mv"]]
        if not day_factors.empty and factor_cols:
            factor_mean = day_factors[factor_cols[:7]].mean().fillna(0).values
        else:
            factor_mean = np.zeros(7)

        # 市场状态
        day_data = self.daily[self.daily["trade_date"] == current_date]
        if not day_data.empty:
            market_ret = day_data["ret"].mean() if "ret" in day_data.columns else 0.0
            market_vol = day_data["ret"].std() if "ret" in day_data.columns else 0.0
            market_range = (day_data["high"].mean() - day_data["low"].mean()) / day_data["close"].mean() if "close" in day_data.columns else 0.0
        else:
            market_ret, market_vol, market_range = 0.0, 0.0, 0.0

        # 持仓状态
        holding_pct = len(self.holdings) / self.n_stocks if self.n_stocks > 0 else 0.0

        obs = np.concatenate([
            factor_mean[:7],
            [market_ret, market_vol, market_range],
            [holding_pct]
        ]).astype(np.float32)

        return obs

    def get_sharpe_ratio(self):
        """获取当前夏普比率"""
        if len(self.returns_history) < 5:
            return 0.0
        ret_arr = np.array(self.returns_history)
        if ret_arr.std() == 0:
            return 0.0
        return ret_arr.mean() / ret_arr.std() * np.sqrt(252)
