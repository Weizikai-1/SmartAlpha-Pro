"""RL与回测集成 - 使用RL模型进行仓位管理"""
import logging

import numpy as np
import pandas as pd

from smartalpha.rl.env import PortfolioEnv, HAS_GYM
from smartalpha.rl.sac_trainer import SACTrainer

logger = logging.getLogger(__name__)


class RLBacktestIntegration:
    """RL增强回测：用SAC模型动态调整仓位"""

    def __init__(self, base_engine=None):
        self.base_engine = base_engine
        self.env = None
        self.rl_model = None

    def run_rl_backtest(self, use_trained_model=True):
        """运行RL增强回测"""
        logger.info("启动RL增强回测...")

        if not HAS_GYM:
            logger.warning("gymnasium未安装，使用随机策略回测")
            use_trained_model = False

        # 初始化环境
        self.env = PortfolioEnv()

        if use_trained_model:
            try:
                trainer = SACTrainer(env=self.env)
                trainer.load_model()
                self.rl_model = trainer
                logger.info("使用已训练SAC模型")
            except FileNotFoundError:
                logger.warning("未找到训练好的模型，将训练新模型...")
                trainer = SACTrainer(env=self.env, total_timesteps=20000)
                trainer.train()
                self.rl_model = trainer
        else:
            logger.info("使用随机策略")

        # 运行回测
        obs, _ = self.env.reset()
        done = False
        nav_history = []

        while not done:
            if self.rl_model:
                action = self.rl_model.get_action(obs)
            else:
                action = np.array([np.random.uniform(-1, 1)])

            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            nav_history.append({
                "step": self.env.current_step,
                "portfolio_value": info["portfolio_value"],
                "daily_return": info["daily_return"],
                "drawdown": info["drawdown"],
                "reward": reward,
            })

        # 计算绩效
        nav_df = pd.DataFrame(nav_history)
        returns = nav_df["daily_return"].dropna()
        if len(returns) > 0:
            # 【P0修复】修正年化收益计算，考虑rebalance_days（持仓周期）
            # 假设rebalance_days=5（每周调仓），每个step代表一个调仓周期
            rebalance_days = 5
            # 正确的年化公式：(1 + 累计收益)^(252/总交易天数) - 1
            # 总交易天数 = 收益周期数 * 调仓周期天数
            total_trading_days = len(returns) * rebalance_days
            ann_ret = (1 + returns).prod() ** (252 / total_trading_days) - 1
            ann_vol = returns.std() * np.sqrt(252 / rebalance_days)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            max_dd = nav_df["drawdown"].max()
        else:
            ann_ret, ann_vol, sharpe, max_dd = 0, 0, 0, 0

        report = {
            "annual_return": round(ann_ret, 4),
            "annual_volatility": round(ann_vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "final_value": round(self.env.portfolio_value, 2),
            "rl_enhanced": HAS_GYM,
        }

        logger.info(f"RL回测完成: 夏普={sharpe:.4f}, 回撤={max_dd:.4f}, 终值={self.env.portfolio_value:,.0f}")
        return report, nav_df
