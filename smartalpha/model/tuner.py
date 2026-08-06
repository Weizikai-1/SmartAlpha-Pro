"""超参数优化 - Optuna自动调参，支持LightGBM和Transformer"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from smartalpha.config import MODEL_SAVE_DIR as MODEL_DIR

logger = logging.getLogger(__name__)

# LightGBM 默认参数
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
}

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.warning("optuna未安装，超参数优化不可用。运行: pip install optuna")


def _time_series_split(dates, n_splits=3, label_horizon=5):
    """
    时间序列交叉验证切分

    【P1修复】引入purge间隔防止标签泄露
    - label_horizon: 标签前瞻窗口长度（默认5天）
    - 在每个fold的train/val切分点前，剔除label_horizon天的训练样本
    - 参考 Qlib/FinRL 的最佳实践，确保训练集不包含验证集标签信息
    """
    unique_dates = sorted(dates.unique())
    fold_size = len(unique_dates) // (n_splits + 1)
    purge_days = label_horizon
    splits = []

    for i in range(1, n_splits + 1):
        split_idx = i * fold_size
        split_date = unique_dates[split_idx]

        # 【P1修复】引入purge间隔
        if split_idx > purge_days:
            purge_idx = split_idx - purge_days
        else:
            purge_idx = split_idx

        split_date_purged = unique_dates[purge_idx]

        # 训练集：使用purge后的切分点
        train_mask = dates < split_date_purged

        # 验证集为切分点后的一段
        val_end = unique_dates[min((i + 1) * fold_size, len(unique_dates) - 1)]
        val_mask = (dates >= split_date) & (dates < val_end)

        splits.append((train_mask, val_mask))

    return splits


class LGBMHyperTuner:
    """LightGBM超参数优化器"""

    def __init__(self, n_trials=50, timeout=3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None
        self.study = None

    def _objective(self, trial, X, y, dates):
        """Optuna优化目标函数：时间序列CV的IC均值"""
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbose": -1,
            "n_jobs": -1,
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        import lightgbm as lgb
        ic_scores = []
        rmse_scores = []

        # 【P1修复】使用带purge的时序切分
        for train_mask, val_mask in _time_series_split(dates, n_splits=3, label_horizon=5):
            if train_mask.sum() < 100 or val_mask.sum() < 50:
                continue
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]

            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params, train_data, num_boost_round=500,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )

            y_pred = model.predict(X_val)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
            # 【P0修复】修复dates_val未定义错误，使用dates[val_mask]
            dates_val = dates[val_mask]
            ic_series = pd.DataFrame({"pred": y_pred, "y": y_val.values, "date": dates_val.values}).groupby("date").apply(lambda g: g["pred"].corr(g["y"]) if len(g) > 2 else np.nan)
            ic = ic_series.mean()
            ic_scores.append(ic)
            rmse_scores.append(rmse)

        if not ic_scores:
            return -999

        # 目标：最大化IC，同时惩罚RMSE过高
        mean_ic = np.mean(ic_scores)
        mean_rmse = np.mean(rmse_scores)
        # 使用IC作为主要优化目标，RMSE作为软约束
        score = mean_ic * 100 - mean_rmse * 10
        return score

    def tune(self, X, y, dates):
        """执行超参数搜索"""
        if not HAS_OPTUNA:
            logger.error("optuna未安装，无法调参")
            return None

        logger.info(f"启动LightGBM超参数优化: trials={self.n_trials} (带purge防护)")
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            study_name="lgbm_tuning"
        )

        self.study.optimize(
            lambda trial: self._objective(trial, X, y, dates),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        logger.info(f"最优参数: {self.best_params}")
        logger.info(f"最优IC: {self.study.best_value:.4f}")

        # 保存参数
        pd.DataFrame([self.best_params]).to_json(MODEL_DIR / "lgbm_best_params.json")
        return self.best_params

    def run_full(self, selected_factors=None):
        """完整流程：准备数据→调参→返回最优参数"""
        from smartalpha.model.lgbm import LGBMTrainer
        trainer = LGBMTrainer()
        X, y, dates, feature_cols = trainer.prepare_data(selected_factors=selected_factors)
        if X is None:
            return None
        best = self.tune(X, y, dates)
        if best:
            # 合并到config的默认参数上
            full_params = dict(LGBM_PARAMS)
            full_params.update(best)
            logger.info(f"调参完成，最优参数已保存")
            return full_params
        return None


class TransformerHyperTuner:
    """Transformer模型超参数优化"""

    def __init__(self, n_trials=20, timeout=3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.best_params = None

    def tune(self, X, y):
        """简化版Transformer调参"""
        if not HAS_OPTUNA:
            logger.error("optuna未安装")
            return None

        logger.info(f"启动Transformer超参数优化: trials={self.n_trials}")

        import torch
        import torch.nn as nn
        from smartalpha.model.transformer import TransformerModel

        def objective(trial):
            params = {
                "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
                "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
                "num_layers": trial.suggest_int("num_layers", 1, 4),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            }
            # 实际训练3个epoch快速评估
            from smartalpha.model.transformer import TransformerTrainer
            trainer = TransformerTrainer(
                input_dim=X.shape[1],
                seq_len=20
            )
            trainer.model = TransformerModel(
                input_dim=X.shape[1],
                d_model=params["d_model"],
                nhead=params["nhead"],
                num_layers=params["num_layers"],
                dropout=params["dropout"]
            ).to(trainer.device)
            optimizer = torch.optim.Adam(trainer.model.parameters(), lr=params["lr"])
            criterion = torch.nn.MSELoss()

            # 准备序列数据
            X_seq, y_seq = trainer.prepare_sequences(X, y)
            if len(X_seq) == 0:
                return -999
            split = int(len(X_seq) * 0.8)
            X_tr, y_tr = X_seq[:split], y_seq[:split]
            X_va, y_va = X_seq[split:], y_seq[split:]

            trainer.model.train()
            for epoch in range(3):
                pred = trainer.model(X_tr)
                loss = criterion(pred, y_tr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            trainer.model.eval()
            with torch.no_grad():
                pred_val = trainer.model(X_va).cpu().numpy()
                y_val = y_va.cpu().numpy()
                ic = np.corrcoef(pred_val, y_val)[0, 1] if len(pred_val) > 2 else 0
            return ic

        study = optuna.create_study(direction="maximize", study_name="transformer_tuning")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        self.best_params = study.best_params
        logger.info(f"Transformer最优参数: {self.best_params}")
        return self.best_params


class AutoTuner:
    """自动调参入口：根据配置选择调参器"""

    def __init__(self, model_type="lgbm"):
        self.model_type = model_type

    def run(self, selected_factors=None):
        if self.model_type == "lgbm":
            tuner = LGBMHyperTuner(n_trials=30)
            return tuner.run_full(selected_factors=selected_factors)
        elif self.model_type == "transformer":
            tuner = TransformerHyperTuner(n_trials=10)
            # Transformer调参需要X,y
            from smartalpha.model.lgbm import LGBMTrainer
            X, y, _, _ = LGBMTrainer().prepare_data(selected_factors=selected_factors)
            if X is not None:
                return tuner.tune(X, y)
        return None
