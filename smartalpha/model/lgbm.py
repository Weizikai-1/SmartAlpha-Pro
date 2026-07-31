"""LightGBM 预测器 — 严格时序分割 + 特征重要性。

设计原则:
- 训练/验证按日期分割，不做随机 shuffle (防止未来信息泄漏)
- Purge 间隔: train结束日与test开始日之间留N天空隙
- 所有评估指标仅基于验证集 (out-of-sample)
- 不虚构高Sharpe，输出真实效果
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


@dataclass
class ModelResult:
    """模型训练/预测结果。"""

    predictions: Optional[pd.Series] = None   # 预测值 (仅包含验证集)
    feature_importance: Optional[pd.Series] = None
    train_scores: dict = field(default_factory=dict)
    oof_predictions: Optional[pd.Series] = None  # 全部OOF预测


class LightGBMPredictor:
    """LightGBM 预测器。

    使用 walk-forward 分割训练，严格保证时序：
    train = [T_start, T_split - purge_days]
    val   = [T_split, T_split + val_days]

    使用示例:
        model = LightGBMPredictor(params={"num_leaves": 31, "n_estimators": 100})
        result = model.train_predict(X, y, dates, train_end="2024-12-31")
        print(result.feature_importance.head(10))
    """

    def __init__(
        self,
        params: Optional[dict] = None,
        early_stopping_rounds: int = 50,
        verbose: int = -1,
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm 未安装，请运行: pip install lightgbm")

        self.params = params or {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
            "verbose": -1,
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self._model: Optional[lgb.Booster] = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # 单次训练 + 预测
    # ------------------------------------------------------------------

    def train_predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.Series,
        train_end: str,
        purge_days: int = 5,
        eval_days: int = 60,
    ) -> ModelResult:
        """单次 walk-forward 训练并预测。

        Args:
            X: 特征矩阵。
            y: 标签（前向收益率）。
            dates: 每行对应的日期（用于时序分割）。
            train_end: 训练集结束日期 (YYYYMMDD)。
            purge_days: train/test 之间的清空间隔天数。
            eval_days: 验证集持续天数。

        Returns:
            ModelResult，含预测值、特征重要性、训练指标。
        """
        dates = pd.to_datetime(dates)
        train_end_dt = pd.to_datetime(train_end)

        # 训练集: 截止 train_end 的所有数据
        train_mask = dates <= train_end_dt
        X_train = X[train_mask]
        y_train = y[train_mask]
        dates_train = dates[train_mask]

        eval_start = train_end_dt + pd.Timedelta(days=purge_days)
        eval_end = eval_start + pd.Timedelta(days=eval_days)
        val_mask = (dates > eval_start) & (dates <= eval_end)
        X_val = X[val_mask]
        y_val = y[val_mask]

        if len(X_train) < 100 or len(X_val) < 10:
            return ModelResult()

        # 内部再做一次时序验证集 (用于 early stopping)
        n_train = int(len(X_train) * 0.8)
        X_tr, X_ev = X_train.iloc[:n_train], X_train.iloc[n_train:]
        y_tr, y_ev = y_train.iloc[:n_train], y_train.iloc[n_train:]

        self._feature_names = list(X.columns)

        # LightGBM Dataset
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_ev, label=y_ev, reference=dtrain)

        # 训练
        self._model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dvalid],
            valid_names=["valid"],
            num_boost_round=self.params.get("n_estimators", 200),
            callbacks=[lgb.early_stopping(self.early_stopping_rounds),
                       lgb.log_evaluation(0)],
        )

        # 预测验证集
        preds = self._model.predict(X_val, num_iteration=self._model.best_iteration)

        # 特征重要性 (gain)
        importance = pd.Series(
            self._model.feature_importance(importance_type="gain"),
            index=self._feature_names,
        ).sort_values(ascending=False)

        # 训练指标
        train_preds = self._model.predict(X_tr, num_iteration=self._model.best_iteration)
        ev_preds = self._model.predict(X_ev, num_iteration=self._model.best_iteration)

        return ModelResult(
            predictions=pd.Series(preds, index=X_val.index),
            feature_importance=importance,
            train_scores={
                "train_rmse": np.sqrt(np.mean((y_tr.values - train_preds) ** 2)),
                "valid_rmse": np.sqrt(np.mean((y_ev.values - ev_preds) ** 2)),
                "best_iteration": self._model.best_iteration,
                "train_samples": len(X_tr),
                "valid_samples": len(X_val),
            },
        )

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """使用已训练模型预测。"""
        if self._model is None:
            raise RuntimeError("模型未训练，请先调用 train_predict()")
        return self._model.predict(X, num_iteration=self._model.best_iteration)

    @property
    def is_trained(self) -> bool:
        return self._model is not None


# 别名: 兼容 ensemble/tuner/auto_retrain 中的 LGBMTrainer 引用
LGBMTrainer = LightGBMPredictor
