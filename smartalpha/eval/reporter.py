"""评估报告生成模块。

将 metrics.py 的输出格式化为人类可读的评估报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FactorReport:
    """因子评估报告。

    Attributes:
        factor_name: 因子名称。
        metrics: evaluate_factor() 的返回字典。
        passed: 是否通过最低标准。
    """

    factor_name: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    passed: bool = False

    # 最低评估标准
    MIN_IC_IR = 0.3
    MIN_SHARPE = 0.5
    MAX_DRAWDOWN = -0.5
    MIN_HIT_RATE = 0.5

    def generate(self) -> str:
        """生成评估报告文本。

        Returns:
            格式化的报告字符串。
        """
        m = self.metrics
        if not m:
            return "空报告（无评估数据）"

        lines = []
        lines.append("=" * 60)
        lines.append(f"因子评估报告: {self.factor_name or m.get('factor_name', '未命名')}")
        lines.append("=" * 60)
        lines.append(f"数据时间: {m.get('data_period', 'N/A')}")
        lines.append(f"数据点数: {m.get('data_points', 'N/A')}")
        lines.append("")

        # IC 统计
        ic = m.get("ic", {})
        if ic:
            lines.append("--- IC 统计 (Pearson) ---")
            for period, stats in ic.items():
                label = "全样本IC" if stats.get("is_full_sample") else "IC均值"
                lines.append(
                    f"  {period}: {label}={stats['ic_mean']:.4f}, "
                    f"IC_IR={stats['ic_ir']:.4f}, "
                    f"胜率={stats['hit_rate']:.0%} "
                    f"({stats['positive_days']}/{stats['total_days']})"
                )

        # Rank IC 统计
        ric = m.get("rank_ic", {})
        if ric:
            lines.append("--- Rank IC 统计 (Spearman) ---")
            for period, stats in ric.items():
                label = "全样本IC" if stats.get("is_full_sample") else "IC均值"
                lines.append(
                    f"  {period}: {label}={stats['ic_mean']:.4f}, "
                    f"IC_IR={stats['ic_ir']:.4f}"
                )

        # 策略指标
        lines.append("")
        lines.append("--- 策略指标 (基于因子值的简单多空) ---")
        lines.append(f"  年化夏普: {m.get('sharpe', 'N/A')}")
        lines.append(f"  年化收益: {m.get('annual_return', 'N/A')}")
        lines.append(f"  最大回撤: {m.get('max_drawdown', 'N/A')}")
        lines.append(f"  日均换手: {m.get('turnover', 'N/A')}")

        # 结论
        lines.append("")
        lines.append("--- 结论 ---")
        self._check_pass()
        lines.append(f"  总体评估: {'✅ 通过' if self.passed else '❌ 未通过'} 最低标准")
        lines.append(f"  评估标准: IC_IR≥{self.MIN_IC_IR}, "
                     f"Sharpe≥{self.MIN_SHARPE}, "
                     f"MDD≥{self.MAX_DRAWDOWN}")

        return "\n".join(lines)

    def _check_pass(self) -> None:
        """根据最低标准判断是否通过。"""
        m = self.metrics
        if not m:
            return

        # 取最短持有期（通常是 ret_1d）的 IC
        ic = m.get("ic", {})
        ic_ir = 0.0
        for stats in ic.values():
            ic_ir = max(ic_ir, abs(stats.get("ic_ir", 0)))

        sharpe = m.get("sharpe", 0) or 0
        mdd = m.get("max_drawdown", 0) or 0

        self.passed = (
            abs(ic_ir) >= self.MIN_IC_IR
            and abs(sharpe) >= self.MIN_SHARPE
            and mdd >= self.MAX_DRAWDOWN
        )
