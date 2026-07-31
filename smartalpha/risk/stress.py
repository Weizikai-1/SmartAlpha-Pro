"""压力测试模块。

模拟4种历史极端场景对组合的冲击：
1. 2015年股灾 (2015-06-12 ~ 2015-07-08, 上证 -32%)
2. 2020年疫情 (2020-01-23 ~ 2020-02-03, 上证 -8%)
3. 2024年2月调整 (2024-02-01 ~ 2024-02-05, 中小盘 -20%)
4. 闪崩场景 (单日 -7% + 连续3天 -2%)

每类场景评估：组合净值变化、最大回撤、VaR突破次数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# 预定义的历史极端场景 (A股)
HISTORICAL_SCENARIOS = {
    "2015股灾": {
        "period": ("20150612", "20150708"),
        "description": "2015年股灾: 杠杆资金踩踏, 上证从5178跌至3507 (-32%)",
        "shock": [0.0000, -0.0300, -0.0200, -0.0350, -0.0300, -0.0700, -0.0550,
                  0.0200, -0.0250, -0.0450, -0.0350, -0.0150, 0.0100, -0.0200,
                  -0.0100, -0.0550, -0.0600, -0.0850, 0.0000, 0.0400, 0.0200],
    },
    "2020疫情": {
        "period": ("20200123", "20200203"),
        "description": "2020年新冠疫情: 春节后首日暴跌, 上证 -7.7%",
        "shock": [-0.0290, -0.0770],
    },
    "2024小微盘": {
        "period": ("20240201", "20240205"),
        "description": "2024年2月小微盘流动性危机: 中证2000连续暴跌, 累计 -20%",
        "shock": [-0.0250, -0.0450, -0.0850],
    },
    "闪崩场景": {
        "period": ("", ""),
        "description": "模拟闪崩: T日 -7%, 连续3日 -2%",
        "shock": [-0.0700, -0.0200, -0.0200, -0.0200],
    },
}


@dataclass
class StressTestResult:
    """单场景压力测试结果。"""

    scenario_name: str = ""
    description: str = ""
    start_nav: float = 1.0
    end_nav: float = 1.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    var_breaches: int = 0          # 突破 VaR 的次数
    worst_day: float = 0.0         # 最差单日收益
    daily_nav: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.scenario_name}: {self.description}\n"
            f"  净值 {self.start_nav:.3f} → {self.end_nav:.3f} "
            f"(收益 {self.total_return:.2%})\n"
            f"  最大回撤 {self.max_drawdown:.2%}, "
            f"最差日 {self.worst_day:.2%}, "
            f"VaR突破 {self.var_breaches}次"
        )


class StressTester:
    """压力测试器。

    使用示例:
        tester = StressTester()
        results = tester.run_all(portfolio_returns, var_95=-0.02)
        for r in results:
            print(r.summary())
    """

    def run_all(
        self,
        portfolio_returns: pd.Series,
        var_95: float = -0.02,
    ) -> list[StressTestResult]:
        """对所有预定义场景执行压力测试。

        Args:
            portfolio_returns: 组合日收益率序列（用于估计波动率）。
            var_95: 组合日常 VaR (95%置信度)，用于统计突破次数。

        Returns:
            各场景测试结果列表。
        """
        results = []
        vol = float(portfolio_returns.std()) if len(portfolio_returns) > 0 else 0.01

        for name, config in HISTORICAL_SCENARIOS.items():
            result = self._run_scenario(name, config, vol, var_95)
            results.append(result)

        return results

    def _run_scenario(
        self,
        name: str,
        config: dict,
        base_vol: float,
        var_95: float,
    ) -> StressTestResult:
        """执行单个场景的压力测试。

        将历史场景的收益率缩放至当前组合的波动率水平：
        scaled_shock[i] = original_shock[i] × (base_vol / scenario_vol)
        """
        shocks = np.array(config["shock"])
        scenario_vol = float(np.std(shocks)) if len(shocks) > 1 else 0.02

        # 缩放到当前组合波动水平
        if scenario_vol > 1e-10 and base_vol > 1e-10:
            scale = base_vol / scenario_vol
            # 限制缩放范围 [0.5, 3.0]，避免极端异常
            scale = max(0.5, min(3.0, scale))
        else:
            scale = 1.0

        scaled = shocks * scale

        # 模拟净值路径
        nav_path = [1.0]
        peak = 1.0
        max_dd = 0.0
        var_breaches = 0
        worst_day = 0.0

        for s in scaled:
            new_nav = nav_path[-1] * (1 + float(s))
            nav_path.append(new_nav)
            peak = max(peak, new_nav)
            dd = (new_nav - peak) / peak
            max_dd = min(max_dd, dd)
            worst_day = min(worst_day, float(s))
            if float(s) < var_95:
                var_breaches += 1

        total_ret = nav_path[-1] - 1.0

        return StressTestResult(
            scenario_name=name,
            description=config["description"],
            start_nav=1.0,
            end_nav=round(nav_path[-1], 4),
            total_return=round(total_ret, 4),
            max_drawdown=round(max_dd, 4),
            var_breaches=var_breaches,
            worst_day=round(worst_day, 4),
            daily_nav=[round(v, 4) for v in nav_path],
        )

    @staticmethod
    def custom_scenario(
        name: str,
        description: str,
        shocks: list[float],
        var_95: float = -0.02,
    ) -> StressTestResult:
        """自定义场景压力测试。

        Args:
            name: 场景名称。
            description: 场景描述。
            shocks: 逐日收益率冲击序列。
            var_95: VaR阈值。
        """
        config = {"period": ("", ""), "description": description, "shock": shocks}
        tester = StressTester()
        return tester._run_scenario(name, config, 0.02, var_95)
