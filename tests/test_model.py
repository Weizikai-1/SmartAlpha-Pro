"""ML模型层单元测试。

测试: LightGBM训练/预测、Walk-Forward滚动训练、Purge防泄漏。

注意: LightGBM 在某些沙箱环境中因 DLL 加载失败无法运行。
在 CI 环境中可以正常通过。
"""

import numpy as np
import pandas as pd
import pytest

# 尝试导入 lightgbm，如果 DLL 被阻止则跳过所有测试
try:
    import lightgbm as _  # noqa: F401
except (ImportError, OSError) as e:
    pytest.skip(f"lightgbm 不可用: {e}", allow_module_level=True)

from smartalpha.model.lgbm import LightGBMPredictor, ModelResult
from smartalpha.model.trainer import WalkForwardTrainer, WalkForwardResult


# ============================================================================
# 生成模拟数据
# ============================================================================

@pytest.fixture
def sample_data():
    """生成带趋势的模拟因子+标签数据 (500天, 5个因子)。"""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-01", periods=n, freq="B")

    f1 = np.random.randn(n) * 0.01
    f2 = np.random.randn(n) * 0.02
    f3 = np.random.randn(n) * 0.015

    noise = np.random.randn(n) * 0.02
    y = 0.001 + 0.3 * f1 + 0.2 * f2 + noise

    X = pd.DataFrame({
        "momentum": f1,
        "volatility": f2,
        "volume_ratio": f3,
        "noise_1": np.random.randn(n),
        "noise_2": np.random.randn(n),
    }, index=dates)

    return X, pd.Series(y, index=dates), pd.Series(dates, index=dates)


# ============================================================================
# LightGBM 训练/预测测试
# ============================================================================

class TestLightGBMPredictor:
    def test_train_predict_basic(self, sample_data):
        X, y, dates = sample_data
        model = LightGBMPredictor(
            params={"num_leaves": 15, "n_estimators": 50, "verbose": -1}
        )
        result = model.train_predict(
            X, y, dates, train_end="2023-06-30", purge_days=5, eval_days=30
        )
        assert result.predictions is not None
        assert len(result.predictions) > 0

    def test_model_is_trained(self, sample_data):
        X, y, dates = sample_data
        model = LightGBMPredictor(
            params={"num_leaves": 15, "n_estimators": 50, "verbose": -1}
        )
        model.train_predict(X, y, dates, train_end="2023-06-30", purge_days=5, eval_days=30)
        assert model.is_trained

    def test_feature_importance(self, sample_data):
        X, y, dates = sample_data
        model = LightGBMPredictor(
            params={"num_leaves": 15, "n_estimators": 100, "verbose": -1}
        )
        result = model.train_predict(X, y, dates, train_end="2023-06-30", purge_days=5, eval_days=60)
        assert result.feature_importance is not None
        assert len(result.feature_importance) == 5

    def test_train_scores(self, sample_data):
        X, y, dates = sample_data
        model = LightGBMPredictor(
            params={"num_leaves": 15, "n_estimators": 50, "verbose": -1}
        )
        result = model.train_predict(X, y, dates, train_end="2023-06-30", purge_days=5, eval_days=30)
        assert "train_rmse" in result.train_scores
        assert "valid_rmse" in result.train_scores
        assert result.train_scores["best_iteration"] > 0

    def test_predict_untrained_raises(self, sample_data):
        X, _, _ = sample_data
        model = LightGBMPredictor()
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_purge_respected(self, sample_data):
        X, y, dates = sample_data
        model = LightGBMPredictor(
            params={"num_leaves": 15, "n_estimators": 50, "verbose": -1}
        )
        result = model.train_predict(
            X, y, dates, train_end="2023-06-30", purge_days=10, eval_days=30
        )
        if result.predictions is not None:
            val_min_date = dates.loc[result.predictions.index].min()
            purge_end = pd.Timestamp("2023-07-10")
            assert val_min_date >= purge_end


# ============================================================================
# Walk-Forward 训练测试
# ============================================================================

class TestWalkForwardTrainer:
    def test_run_basic(self, sample_data):
        X, y, dates = sample_data
        trainer = WalkForwardTrainer(
            purge_days=5, val_days=30, step_days=60,
            lgbm_params={"num_leaves": 15, "n_estimators": 50, "verbose": -1},
        )
        result = trainer.run(X, y, dates, min_train_days=200)
        assert result.oof_predictions is not None
        assert len(result.oof_predictions) > 0
        assert result.metrics["n_folds"] >= 1

    def test_oof_only(self, sample_data):
        X, y, dates = sample_data
        trainer = WalkForwardTrainer(
            purge_days=5, val_days=30, step_days=60,
            lgbm_params={"num_leaves": 15, "n_estimators": 50, "verbose": -1},
        )
        result = trainer.run(X, y, dates, min_train_days=200)
        assert len(result.oof_predictions) < len(y)

    def test_metrics_valid(self, sample_data):
        X, y, dates = sample_data
        trainer = WalkForwardTrainer(
            purge_days=5, val_days=30, step_days=60,
            lgbm_params={"num_leaves": 15, "n_estimators": 50, "verbose": -1},
        )
        result = trainer.run(X, y, dates, min_train_days=200)
        assert "ic" in result.metrics
        assert "rank_ic" in result.metrics
        assert "rmse" in result.metrics
        assert result.metrics["ic"] > 0

    def test_feature_importance_averaged(self, sample_data):
        X, y, dates = sample_data
        trainer = WalkForwardTrainer(
            purge_days=5, val_days=30, step_days=60,
            lgbm_params={"num_leaves": 15, "n_estimators": 50, "verbose": -1},
        )
        result = trainer.run(X, y, dates, min_train_days=200)
        if result.feature_importance is not None and len(result.feature_importance) > 0:
            assert not result.feature_importance.isna().any()

    def test_insufficient_data(self, sample_data):
        X, y, dates = sample_data
        trainer = WalkForwardTrainer()
        result = trainer.run(X, y, dates, min_train_days=1000)
        assert result.oof_predictions is None
