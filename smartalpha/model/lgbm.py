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

        # 训练 (lightgbm 4.x + Python 3.13 兼容: 避免 early_stopping 的 CVBooster bug)
        evals_result = {}
        num_boost_round = self.params.get("n_estimators", 200)
        self._model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dvalid],
            valid_names=["valid"],
            num_boost_round=num_boost_round,
            callbacks=[lgb.record_evaluation(evals_result),
                       lgb.log_evaluation(0)],
        )

        # 最佳迭代 (从 evals_result 手动计算)
        if evals_result and "valid" in evals_result and evals_result["valid"]:
            valid_metric = list(evals_result["valid"].values())[0]
            best_iteration = int(np.argmin(valid_metric)) + 1
            best_iteration = max(1, min(best_iteration, len(valid_metric)))
        else:
            best_iteration = num_boost_round

        # 预测验证集
        preds = self._model.predict(X_val, num_iteration=best_iteration)

        # 特征重要性 (gain)
        importance = pd.Series(
            self._model.feature_importance(importance_type="gain"),
            index=self._feature_names,
        ).sort_values(ascending=False)

        # 训练指标
        train_preds = self._model.predict(X_tr, num_iteration=best_iteration)
        ev_preds = self._model.predict(X_ev, num_iteration=best_iteration)

        return ModelResult(
            predictions=pd.Series(preds, index=X_val.index),
            feature_importance=importance,
            train_scores={
                "train_rmse": np.sqrt(np.mean((y_tr.values - train_preds) ** 2)),
                "valid_rmse": np.sqrt(np.mean((y_ev.values - ev_preds) ** 2)),
                "best_iteration": best_iteration,
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

    # ------------------------------------------------------------------
    # 便捷方法: 兼容 ensemble / auto_retrain / tuner
    # ------------------------------------------------------------------

    def prepare_data(self, selected_factors=None):
        """加载因子数据，计算前瞻收益标签，返回 (X, y, dates, feature_cols)。

        使用 daily_masked.parquet 计算 fwd_ret_5d 作为标签。

        Args:
            selected_factors: 因子列名列表，None 则自动排除元数据列。

        Returns:
            (X: DataFrame, y: Series, dates: Series, feature_cols: list)
        """
        import pandas as pd
        from smartalpha.config import PROCESSED_DIR

        def _read(p):
            try:
                return pd.read_parquet(p)
            except OSError:
                return pd.read_parquet(p, engine="fastparquet")

        factors = _read(PROCESSED_DIR / "factors_neutral.parquet")
        daily = _read(PROCESSED_DIR / "daily_masked.parquet")

        # 计算前瞻收益标签
        daily["trade_date"] = pd.to_datetime(daily["trade_date"], format="%Y%m%d")
        daily = daily.sort_values(["ts_code", "trade_date"])
        daily["fwd_ret_5d"] = daily.groupby("ts_code")["close"].pct_change(5).shift(-5)

        factors["trade_date"] = pd.to_datetime(factors["trade_date"], format="%Y%m%d")
        factors = factors.merge(
            daily[["ts_code", "trade_date", "fwd_ret_5d"]],
            on=["ts_code", "trade_date"], how="inner"
        )
        factors = factors.dropna(subset=["fwd_ret_5d"])

        exclude = {"ts_code", "trade_date", "industry", "circ_mv", "log_mv", "fwd_ret_5d"}
        if selected_factors is None:
            selected_factors = [c for c in factors.columns if c not in exclude]

        X = factors[list(selected_factors)]
        y = factors["fwd_ret_5d"]
        dates = factors["trade_date"]

        return X, y, dates, list(selected_factors)

    def train(self, X, y, dates, test_ratio=0.2):
        """快速训练：时序分割后调用 train_predict。

        Args:
            X: 特征矩阵 (DataFrame 或 ndarray)。
            y: 标签序列。
            dates: 每行对应日期。
            test_ratio: 验证集比例 (0~1)。

        Side-effect: 设置 self.model 指向已训练模型。
        """
        import pandas as pd
        dates_dt = pd.to_datetime(dates)
        unique_dates = sorted(dates_dt.unique())
        split_idx = max(1, min(int(len(unique_dates) * (1 - test_ratio)),
                               len(unique_dates) - 1))
        split_date = unique_dates[split_idx]

        # 确保 X 是 DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

        self.train_predict(X, y, dates, train_end=split_date.strftime("%Y%m%d"))
        self.model = self._model

    def run(self, selected_factors=None):
        """完整流程：准备数据 → 训练 → 评估，返回结果字典。

        Args:
            selected_factors: 因子列名列表。

        Returns:
            {"rmse": float, "ic": float} 或 None。
        """
        import numpy as np
        import pandas as pd

        X, y, dates, _ = self.prepare_data(selected_factors=selected_factors)

        dates_dt = pd.to_datetime(dates)
        unique_dates = sorted(dates_dt.unique())
        split_idx = max(1, len(unique_dates) * 2 // 3)
        train_end = unique_dates[min(split_idx, len(unique_dates) - 1)]

        result = self.train_predict(X, y, dates, train_end=train_end.strftime("%Y%m%d"))
        self.model = self._model

        predictions = result.predictions
        y_val = y.loc[predictions.index]
        ic = predictions.corr(y_val) if len(predictions) > 1 else 0.0
        rmse = np.sqrt(np.mean((predictions.values - y_val.values) ** 2))

        return {"rmse": float(rmse), "ic": float(ic)}

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        """已训练模型（兼容 ensemble 的 self.lgbm_model = trainer.model）。"""
        return self._model

    @model.setter
    def model(self, value):
        self._model = value


# 别名: 兼容 ensemble/tuner/auto_retrain 中的 LGBMTrainer 引用
LGBMTrainer = LightGBMPredictor
