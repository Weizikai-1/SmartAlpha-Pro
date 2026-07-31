"""Walk-Forward 滚动训练管道。

严格时序验证:
- 使用 expanding window 滚动训练
- 每步向前滚动 val_days + purge_days
- 所有预测均为 out-of-sample (OOF)
- 最终 IC/RMSE 仅在验证集上计算
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .lgbm import LightGBMPredictor, ModelResult


@dataclass
class WalkForwardResult:
    """滚动训练完整结果。"""

    oof_predictions: pd.Series | None = None      # 全量OOF预测
    oof_dates: pd.DatetimeIndex | None = None     # OOF对应的日期
    feature_importance: pd.Series | None = None    # 平均特征重要性
    fold_results: list[ModelResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class WalkForwardTrainer:
    """Walk-Forward 滚动训练器。

    每步:
    1. 用 [data_start, train_end] 的数据训练
    2. 在 [train_end + purge, train_end + purge + val_days] 上预测
    3. train_end 前进 val_days 步长
    4. 所有预测拼接为完整 OOF

    使用示例:
        trainer = WalkForwardTrainer(purge_days=5, val_days=60, step_days=60)
        result = trainer.run(X, y, dates, min_train_days=252)
        print(result.metrics)
    """

    def __init__(
        self,
        purge_days: int = 5,
        val_days: int = 60,
        step_days: int = 60,
        lgbm_params: dict | None = None,
    ):
        self.purge_days = purge_days
        self.val_days = val_days
        self.step_days = step_days
        self.lgbm_params = lgbm_params or {}

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        min_train_days: int = 252,
    ) -> WalkForwardResult:
        """执行滚动训练。

        Args:
            X: 特征矩阵 (样本 × 因子)。
            y: 标签 (前向收益率)。
            dates: 每行对应的日期序列。
            min_train_days: 最少训练天数 (默认1年)。

        Returns:
            WalkForwardResult，含全量OOF预测和评估指标。
        """
        dates = pd.to_datetime(dates)
        sorted_idx = dates.argsort()
        dates_sorted = dates.iloc[sorted_idx]
        X_sorted = X.iloc[sorted_idx]
        y_sorted = y.iloc[sorted_idx]

        unique_dates = sorted(dates_sorted.unique())
        if len(unique_dates) < min_train_days + self.val_days:
            return WalkForwardResult()

        all_preds = []
        all_dates = []
        fold_results = []
        importances = []

        # 确定滚动窗口的 train_end 日期列表
        first_train_end_idx = min_train_days
        train_end_indices = list(range(
            first_train_end_idx,
            len(unique_dates) - self.val_days,
            self.step_days,
        ))

        for te_idx in train_end_indices:
            train_end_str = unique_dates[te_idx].strftime("%Y%m%d")

            try:
                predictor = LightGBMPredictor(params=self.lgbm_params)
                result = predictor.train_predict(
                    X_sorted,
                    y_sorted,
                    dates_sorted,
                    train_end=train_end_str,
                    purge_days=self.purge_days,
                    eval_days=self.val_days,
                )
            except Exception:
                import logging
                logging.getLogger(__name__).warning(f"Fold train_end={train_end_str} 训练失败", exc_info=True)
                continue

            if result.predictions is not None and len(result.predictions) > 0:
                all_preds.append(result.predictions)
                all_dates.extend(dates_sorted.loc[result.predictions.index])
                fold_results.append(result)
                if result.feature_importance is not None:
                    importances.append(result.feature_importance)

        if not all_preds:
            return WalkForwardResult()

        # 拼接 OOF 预测
        oof_preds = pd.concat(all_preds)
        oof_dates = pd.DatetimeIndex(all_dates)

        # 平均特征重要性
        if importances:
            avg_importance = (
                pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
            )
        else:
            avg_importance = pd.Series(dtype=float)

        # 计算评估指标 (仅在OOF上)
        # 对齐预测值和真实值
        common_idx = oof_preds.index.intersection(y_sorted.index)
        pred_aligned = oof_preds.loc[common_idx]
        true_aligned = y_sorted.loc[common_idx]

        metrics = {
            "n_folds": len(fold_results),
            "n_oof_samples": len(pred_aligned),
            "rmse": float(np.sqrt(np.mean((pred_aligned.values - true_aligned.values) ** 2))),
            "ic": float(pred_aligned.corr(true_aligned)),
            "rank_ic": float(pred_aligned.rank().corr(true_aligned.rank())),
            "hit_rate": float((np.sign(pred_aligned) == np.sign(true_aligned)).mean()),
        }

        return WalkForwardResult(
            oof_predictions=oof_preds,
            oof_dates=oof_dates,
            feature_importance=avg_importance,
            fold_results=fold_results,
            metrics=metrics,
        )
