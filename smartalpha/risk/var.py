"""VaR/CVaR 计算模块。

三种方法:
1. 历史模拟法 (Historical): 直接用历史收益率分位数
2. 参数法 (Parametric): 假设正态分布, VaR = μ + σ × z_α
3. 蒙特卡洛 (Monte Carlo): 从拟合分布中采样模拟

CVaR (Conditional VaR): 尾部条件期望 = E[loss | loss > VaR]
使用 tail mean 方法，非逐行迭代。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class VaRResult:
    """VaR/CVaR 计算结果。"""

    var_95: float   # 95% 置信度 VaR (日度)
    var_99: float   # 99% 置信度 VaR
    cvar_95: float  # 95% CVaR
    cvar_99: float  # 99% CVaR
    method: str = ""


class VaRCalculator:
    """VaR/CVaR 计算器。

    使用示例:
        calc = VaRCalculator()
        result = calc.historical(daily_returns)
        print(f"VaR95={result.var_95:.4%}, CVaR95={result.cvar_95:.4%}")
    """

    # ------------------------------------------------------------------
    # 历史模拟法
    # ------------------------------------------------------------------

    @staticmethod
    def historical(returns: pd.Series) -> VaRResult:
        """历史模拟法计算 VaR/CVaR。

        直接使用历史收益率的分位数，不做分布假设。
        """
        ret_clean = returns.dropna().values
        if len(ret_clean) < 20:
            return VaRResult(0, 0, 0, 0, "historical")

        var_95 = float(np.percentile(ret_clean, 5))
        var_99 = float(np.percentile(ret_clean, 1))

        # CVaR = 尾部条件期望 (tail mean, 非逐行迭代)
        cvar_95 = float(ret_clean[ret_clean <= var_95].mean()) if np.any(ret_clean <= var_95) else var_95
        cvar_99 = float(ret_clean[ret_clean <= var_99].mean()) if np.any(ret_clean <= var_99) else var_99

        return VaRResult(
            var_95=round(var_95, 6),
            var_99=round(var_99, 6),
            cvar_95=round(cvar_95, 6),
            cvar_99=round(cvar_99, 6),
            method="historical",
        )

    # ------------------------------------------------------------------
    # 参数法 (正态分布假设)
    # ------------------------------------------------------------------

    @staticmethod
    def parametric(returns: pd.Series) -> VaRResult:
        """参数法计算 VaR/CVaR。

        假设收益率服从正态分布 N(μ, σ²)。
        VaR_α = μ + σ × z_α
        CVaR_α = μ + σ × φ(z_α) / (1-α)
        """
        from scipy.stats import norm

        ret_clean = returns.dropna().values
        if len(ret_clean) < 20:
            return VaRResult(0, 0, 0, 0, "parametric")

        mu = float(np.mean(ret_clean))
        sigma = float(np.std(ret_clean, ddof=1))
        if sigma < 1e-10:
            return VaRResult(0, 0, 0, 0, "parametric")

        z_95 = norm.ppf(0.05)   # -1.645
        z_99 = norm.ppf(0.01)   # -2.326

        var_95 = mu + sigma * z_95
        var_99 = mu + sigma * z_99

        # CVaR 公式: μ + σ × φ(z) / (1-α)
        cvar_95 = mu + sigma * norm.pdf(z_95) / 0.05
        cvar_99 = mu + sigma * norm.pdf(z_99) / 0.01

        return VaRResult(
            var_95=round(var_95, 6),
            var_99=round(var_99, 6),
            cvar_95=round(cvar_95, 6),
            cvar_99=round(cvar_99, 6),
            method="parametric",
        )

    # ------------------------------------------------------------------
    # 蒙特卡洛模拟
    # ------------------------------------------------------------------

    @staticmethod
    def monte_carlo(
        returns: pd.Series,
        n_simulations: int = 100_000,
        random_state: int = 42,
    ) -> VaRResult:
        """蒙特卡洛模拟计算 VaR/CVaR。

        从估计的分布中采样 n_simulations 条路径，取分位数。

        Args:
            returns: 历史日收益率。
            n_simulations: 模拟次数。
            random_state: 随机种子（可复现）。
        """
        ret_clean = returns.dropna().values
        if len(ret_clean) < 20:
            return VaRResult(0, 0, 0, 0, "monte_carlo")

        mu = float(np.mean(ret_clean))
        sigma = float(np.std(ret_clean, ddof=1))
        if sigma < 1e-10:
            return VaRResult(0, 0, 0, 0, "monte_carlo")

        rng = np.random.RandomState(random_state)
        simulated = rng.normal(mu, sigma, n_simulations)

        var_95 = float(np.percentile(simulated, 5))
        var_99 = float(np.percentile(simulated, 1))
        cvar_95 = float(simulated[simulated <= var_95].mean())
        cvar_99 = float(simulated[simulated <= var_99].mean())

        return VaRResult(
            var_95=round(var_95, 6),
            var_99=round(var_99, 6),
            cvar_95=round(cvar_95, 6),
            cvar_99=round(cvar_99, 6),
            method="monte_carlo",
        )

    # ------------------------------------------------------------------
    # 一站式计算
    # ------------------------------------------------------------------

    @classmethod
    def all_methods(cls, returns: pd.Series) -> dict[str, VaRResult]:
        """三种方法一起计算，便于对比。"""
        return {
            "historical": cls.historical(returns),
            "parametric": cls.parametric(returns),
            "monte_carlo": cls.monte_carlo(returns),
        }
