"""SAC训练器 - 稳定基线3实现"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smartalpha.rl.env import PortfolioEnv, HAS_GYM

logger = logging.getLogger(__name__)

# 模型存储路径
MODEL_DIR = Path(__file__).parent.parent / "model" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if HAS_GYM:
    from stable_baselines3 import SAC
else:
    SAC = None
    logger.warning("stable-baselines3未安装，SAC训练不可用")


class SACTrainer:
    """Soft Actor-Critic 强化学习训练"""

    def __init__(self, env=None, total_timesteps=50000):
        self.env = env or PortfolioEnv()
        self.total_timesteps = total_timesteps
        self.model = None
        if not HAS_GYM:
            logger.error("gymnasium未安装，无法初始化SAC")

    def train(self):
        """训练SAC模型"""
        if not HAS_GYM or SAC is None:
            logger.error("依赖缺失，跳过SAC训练")
            return None

        logger.info("开始SAC训练...")
        self.model = SAC(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            buffer_size=10000,
            batch_size=64,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            verbose=0,
        )
        self.model.learn(total_timesteps=self.total_timesteps, progress_bar=True)
        model_path = MODEL_DIR / "sac_portfolio.zip"
        self.model.save(model_path)
        logger.info(f"SAC模型保存: {model_path}")
        return self.model

    def evaluate(self, n_episodes=5):
        """评估模型"""
        if self.model is None:
            logger.error("模型未训练")
            return None
        results = []
        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            sharpe = self.env.get_sharpe_ratio()
            results.append({
                "episode": ep + 1,
                "final_value": self.env.portfolio_value,
                "sharpe_ratio": sharpe,
                "total_reward": episode_reward,
            })
            logger.info(f"Episode {ep+1}: 净值={self.env.portfolio_value:,.0f}, 夏普={sharpe:.4f}")
        return pd.DataFrame(results)

    def load_model(self, path=None):
        if not HAS_GYM or SAC is None:
            raise FileNotFoundError("依赖缺失")
        path = path or MODEL_DIR / "sac_portfolio.zip"
        self.model = SAC.load(path, env=self.env)
        logger.info(f"SAC模型加载: {path}")
        return self.model

    def get_action(self, obs):
        """获取动作"""
        if self.model is None:
            return np.array([0.0])
        action, _ = self.model.predict(obs, deterministic=True)
        return action

    def run(self):
        """完整流程：训练→评估"""
        self.train()
        return self.evaluate()
