"""模型监控 - 性能退化检测+数据漂移检测+自动重训触发"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smartalpha.config import MODEL_SAVE_DIR as MODEL_DIR, DATA_DIR, PROCESSED_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


class ModelMonitor:
    """模型监控：检测IC衰减+数据漂移+自动重训触发"""

    def __init__(self):
        self.baseline_ic = None
        self.ic_history = []
        self.baseline_features = None

    def check_performance_degradation(self, current_ic, threshold=-0.3):
        """检测模型性能是否退化"""
        if self.baseline_ic is None:
            self.baseline_ic = current_ic
            logger.info(f"基准IC: {current_ic:.4f}")
            return False

        delta = current_ic - self.baseline_ic
        self.ic_history.append({"ic": current_ic, "delta": delta})

        if delta < threshold:
            logger.warning(f"模型性能退化! IC从{self.baseline_ic:.4f}降至{current_ic:.4f} (变化{delta:+.4f})")
            return True
        else:
            logger.info(f"模型性能正常: IC={current_ic:.4f} (变化{delta:+.4f})")
            return False

    def check_data_drift(self, current_features, baseline_features=None):
        """检测数据漂移"""
        if baseline_features is None:
            self.baseline_features = current_features.describe()
            logger.info("记录数据基准...")
            return False

        current_stats = current_features.describe()
        drift_score = 0
        drifted_cols = []
        for col in current_features.columns:
            if col in baseline_features.columns:
                mean_delta = abs(current_stats[col]["mean"] - baseline_features[col]["mean"])
                base_std = baseline_features[col]["std"]
                if base_std > 0 and mean_delta / base_std > 2.0:
                    drift_score += 1
                    drifted_cols.append(f"{col}({mean_delta/base_std:.1f}σ)")

        if drift_score > 0:
            logger.warning(f"数据漂移检测: {drift_score}个特征异常: {drifted_cols[:5]}")
            return True
        return False

    def check_factor_exposure(self, portfolio_weights, factor_df, factor_cols, limits=None):
        """检查因子暴露是否超限"""
        if limits is None:
            limits = {"max_exposure": 2.0, "min_exposure": -2.0}
        exposures = {}
        for col in factor_cols:
            weighted_exp = 0
            for code, weight in portfolio_weights.items():
                if code in factor_df.index and col in factor_df.columns:
                    weighted_exp += weight * factor_df.loc[code, col]
            exposures[col] = weighted_exp

        violations = []
        for col, exp in exposures.items():
            if exp > limits["max_exposure"] or exp < limits["min_exposure"]:
                violations.append(f"{col}: {exp:.2f}")
        if violations:
            logger.warning(f"因子暴露超限: {violations[:5]}")
        return exposures, violations

    def run_full_check(self):
        """运行完整监控检查"""
        logger.info("=" * 40)
        logger.info("模型监控检查")
        logger.info("=" * 40)

        # 加载回测结果检查绩效
        try:
            result = pd.read_parquet(RESULTS_DIR / "backtest_result.parquet")
            returns = result["daily_return"].dropna()
            if len(returns) > 20:
                volatility = returns.std() * np.sqrt(252)
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                max_dd = (1 + returns).cumprod()
                peak = max_dd.expanding().max()
                dd = ((max_dd - peak) / peak).min()
                logger.info(f"当前年化波动: {volatility:.2%}, 夏普: {sharpe:.2f}, 最大回撤: {dd:.2%}")
        except FileNotFoundError:
            logger.warning("无回测结果")

        # 加载因子数据检查漂移
        try:
            factors = pd.read_parquet(PROCESSED_DIR / "factors_neutral.parquet")
            factor_cols = [c for c in factors.columns if c not in ["ts_code", "trade_date", "industry", "circ_mv", "log_mv"]]
            recent = factors[factor_cols].tail(100)
            baseline = factors[factor_cols].head(100)
            drift = self.check_data_drift(recent, baseline)
            if drift:
                logger.warning("检测到数据漂移，建议重新训练模型")
        except FileNotFoundError:
            logger.warning("无因子数据")

        logger.info("模型监控检查完成")
        return True
