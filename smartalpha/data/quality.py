"""数据质量检查 — 保证加载的数据可用。

检查项:
1. 必要列完整性
2. 缺失值比例
3. 日期连续性
4. 价格合理性
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class QualityReport:
    """数据质量报告。"""

    passed: bool
    checks: dict
    warnings: List[str]

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"DataQuality: {status}"]
        for name, result in self.checks.items():
            lines.append(f"  {name}: {result}")
        for w in self.warnings:
            lines.append(f"  WARN: {w}")
        return "\n".join(lines)


class DataQualityChecker:
    """A股日线数据质量检查器。

    使用示例:
        checker = DataQualityChecker()
        report = checker.check(df)
        if not report.passed:
            print(report)
    """

    # 日线数据必要列
    REQUIRED_COLS = ["open", "high", "low", "close", "vol"]

    # 合理价格范围 (A股)
    MIN_PRICE = 0.01
    MAX_PRICE = 10000.0

    # 允许的最大缺失比例
    MAX_MISSING_RATIO = 0.1

    def check(self, df: pd.DataFrame) -> QualityReport:
        """对日线数据执行全部质量检查。

        Args:
            df: 日线行情DataFrame。

        Returns:
            QualityReport 质量报告。
        """
        checks = {}
        warnings = []

        checks["columns"] = self._check_columns(df, warnings)
        checks["missing"] = self._check_missing(df, warnings)
        checks["price_range"] = self._check_prices(df, warnings)
        checks["date_sorted"] = self._check_date_sorted(df, warnings)

        passed = all(v for v in checks.values())
        return QualityReport(passed=passed, checks=checks, warnings=warnings)

    def _check_columns(
        self, df: pd.DataFrame, warnings: List[str]
    ) -> bool:
        """检查必要列是否存在。"""
        missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
        if missing:
            warnings.append(f"缺少必要列: {missing}")
            return False
        return True

    def _check_missing(
        self, df: pd.DataFrame, warnings: List[str]
    ) -> bool:
        """检查缺失值比例。"""
        if df.empty:
            warnings.append("数据为空")
            return False
        cols = [c for c in self.REQUIRED_COLS if c in df.columns]
        if not cols:
            return True
        if df[cols].empty:
            return False
        missing_ratio = df[cols].isnull().mean().max()
        if pd.isna(missing_ratio) or missing_ratio > self.MAX_MISSING_RATIO:
            warnings.append(f"缺失值比例 {missing_ratio:.1%} 超过阈值 {self.MAX_MISSING_RATIO:.0%}")
            return False
        return True

    def _check_prices(
        self, df: pd.DataFrame, warnings: List[str]
    ) -> bool:
        """检查价格是否在合理范围。"""
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                continue
            too_low = (df[col] < self.MIN_PRICE).sum()
            too_high = (df[col] > self.MAX_PRICE).sum()
            if too_low > 0 or too_high > 0:
                warnings.append(
                    f"{col}: {too_low}条 < {self.MIN_PRICE}, {too_high}条 > {self.MAX_PRICE}"
                )
                return False
        return True

    def _check_date_sorted(
        self, df: pd.DataFrame, warnings: List[str]
    ) -> bool:
        """检查日期是否按升序排列。"""
        if "trade_date" not in df.columns:
            return True
        dates = pd.to_datetime(df["trade_date"])
        if not dates.is_monotonic_increasing:
            warnings.append("日期未按升序排列")
            return False
        return True
