"""RL SAC 训练脚本 — 产出夏普比率指标"""
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

from smartalpha.rl.sac_trainer import SACTrainer
from smartalpha.config import RL_TOTAL_TIMESTEPS

print("=" * 50)
print("RL SAC 训练开始")
print(f"总步数: {RL_TOTAL_TIMESTEPS}")
print("=" * 50)

trainer = SACTrainer(total_timesteps=RL_TOTAL_TIMESTEPS)
trainer.train()
results = trainer.evaluate(n_episodes=5)

if results is not None:
    print()
    print("=" * 50)
    print("训练评估结果")
    print("=" * 50)
    print(results.to_string(index=False))
    print()
    print(f"平均夏普比率: {results['sharpe_ratio'].mean():.4f}")
    print(f"平均终值: {results['final_value'].mean():,.0f}")
else:
    print("训练失败")
