"""模型层测试 — 使用 mock 覆盖 LightGBM（本机 DLL 不可用时替代）。

测试: LightGBMPredictor 初始化/参数、WalkForwardTrainer 流程逻辑。
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# 在 import smartalpha.model 之前 mock lightgbm
lgb_mock = MagicMock()
lgb_mock.Dataset = MagicMock(return_value=MagicMock())
lgb_mock.train = MagicMock()
lgb_mock.early_stopping = MagicMock(return_value=lambda env: None)
lgb_mock.log_evaluation = MagicMock(return_value=lambda env: None)
sys.modules["lightgbm"] = lgb_mock
sys.modules["lightgbm.engine"] = lgb_mock
sys.modules["lightgbm.basic"] = lgb_mock
sys.modules["lightgbm.compat"] = lgb_mock
sys.modules["lightgbm.libpath"] = lgb_mock

import smartalpha.model.lgbm as lgbm_mod
lgbm_mod.HAS_LIGHTGBM = True

from smartalpha.model.lgbm import LightGBMPredictor, ModelResult
from smartalpha.model.trainer import WalkForwardTrainer, WalkForwardResult


# ---------------------------------------------------------------------------
# 辅助函数：创建匹配输入大小的 mock predict
# ---------------------------------------------------------------------------

def _make_predict_side_effect(expected_count):
    """创建 predict 的 side_effect，返回与第一参数行数匹配的预测值。"""
    def _predict(X, *args, **kwargs):
        n = len(X) if hasattr(X, '__len__') else 1
        return np.random.RandomState(42 + n).randn(n) * 0.02
    return _predict


def _make_mock_booster(n_features=3):
    """创建模拟 LightGBM Booster。"""
    mock = MagicMock()
    mock.best_iteration = 50
    mock.predict.side_effect = _make_predict_side_effect(0)
    mock.feature_importance.return_value = np.array([100.0, 50.0, 30.0][:n_features])
    return mock


# ============================================================================
# LightGBMPredictor 测试
# ============================================================================

class TestLightGBMPredictor:
    @pytest.fixture(autouse=True)
    def setup_mock(self):
        """每个测试前重置 lgb mock。"""
        self._booster = _make_mock_booster()
        lgb_mock.train.reset_mock()
        lgb_mock.train.return_value = self._booster
        yield

    @pytest.fixture
    def Xy(self):
        np.random.seed(42)
        dates = pd.date_range("2024-01-02", periods=500, freq="B")
        X = pd.DataFrame({
            "momentum": np.random.randn(500),
            "volume_ratio": np.random.randn(500),
            "rsi": np.random.randn(500),
        })
        y = pd.Series(np.random.randn(500) * 0.02 + 0.001)
        return X, y, pd.Series(dates)

    def test_init_default_params(self):
        model = LightGBMPredictor()
        assert model.params["objective"] == "regression"
        assert model.early_stopping_rounds == 50
        assert not model.is_trained

    def test_init_custom_params(self):
        model = LightGBMPredictor(
            params={"objective": "binary", "num_leaves": 63},
            early_stopping_rounds=20,
            verbose=0,
        )
        assert model.params["objective"] == "binary"
        assert model.params["num_leaves"] == 63
        assert model.early_stopping_rounds == 20

    def test_train_predict_returns_result(self, Xy):
        X, y, dates = Xy
        model = LightGBMPredictor()
        result = model.train_predict(X, y, dates, train_end="2024-10-31", purge_days=5, eval_days=60)
        assert isinstance(result, ModelResult)
        assert result.predictions is not None

    def test_insufficient_data(self, Xy):
        X, y, dates = Xy
        model = LightGBMPredictor()
        result = model.train_predict(X, y, dates, train_end="2024-01-15", purge_days=5, eval_days=10)
        assert result.predictions is None

    def test_is_trained(self, Xy):
        X, y, dates = Xy
        model = LightGBMPredictor()
        assert not model.is_trained
        model.train_predict(X, y, dates, train_end="2024-10-31")
        assert model.is_trained

    def test_predict_untrained_raises(self):
        model = LightGBMPredictor()
        with pytest.raises(RuntimeError, match="未训练"):
            model.predict(pd.DataFrame({"x": [1]}))

    def test_predict_after_training(self, Xy):
        X, y, dates = Xy
        model = LightGBMPredictor()
        model.train_predict(X, y, dates, train_end="2024-10-31")
        preds = model.predict(X.iloc[:10])
        assert isinstance(preds, np.ndarray)

    def test_model_result_fields(self):
        mr = ModelResult()
        assert mr.predictions is None
        assert mr.feature_importance is None
        assert mr.train_scores == {}
        assert mr.oof_predictions is None

    def test_feature_importance(self, Xy):
        X, y, dates = Xy
        model = LightGBMPredictor()
        result = model.train_predict(X, y, dates, train_end="2024-10-31")
        assert result.feature_importance is not None
        assert len(result.feature_importance) == 3

    def test_train_scores(self, Xy):
        X, y, dates = Xy
        model = LightGBMPredictor()
        result = model.train_predict(X, y, dates, train_end="2024-10-31")
        assert "train_rmse" in result.train_scores
        assert "valid_rmse" in result.train_scores
        assert "best_iteration" in result.train_scores


# ============================================================================
# WalkForwardTrainer 测试
# ============================================================================

class TestWalkForwardTrainer:
    @pytest.fixture
    def Xy_long(self):
        np.random.seed(42)
        dates = pd.date_range("2022-01-01", periods=756, freq="B")
        X = pd.DataFrame({
            "f1": np.random.randn(756),
            "f2": np.random.randn(756),
            "f3": np.random.randn(756),
        })
        y = pd.Series(np.random.randn(756) * 0.02)
        return X, y, pd.Series(dates)

    def test_init_defaults(self):
        trainer = WalkForwardTrainer()
        assert trainer.purge_days == 5
        assert trainer.val_days == 60
        assert trainer.step_days == 60

    def test_run_basic(self, Xy_long):
        X, y, dates = Xy_long
        trainer = WalkForwardTrainer(purge_days=5, val_days=60, step_days=120,
                                     lgbm_params={"num_leaves": 15})

        call_count = [0]  # 用列表实现闭包计数器

        def _mock_train_predict(self, X, y, dates, train_end, purge_days, eval_days):
            n = call_count[0]
            call_count[0] += 1
            # 每个 fold 返回不同索引范围，避免重叠
            start = 50 * n
            idx = X.index[start:start + 50]
            return ModelResult(
                predictions=pd.Series(np.random.randn(50) * 0.02, index=idx),
                feature_importance=pd.Series([0.5, 0.3, 0.2], index=["f1", "f2", "f3"]),
                train_scores={"train_rmse": 0.03},
            )

        with patch.object(LightGBMPredictor, "train_predict", _mock_train_predict):
            result = trainer.run(X, y, dates, min_train_days=252)
            assert isinstance(result, WalkForwardResult)
            assert result.metrics["n_folds"] > 0

    def test_insufficient_data(self):
        trainer = WalkForwardTrainer()
        X = pd.DataFrame({"f1": [1, 2]})
        y = pd.Series([0.01, 0.02])
        dates = pd.Series(["2024-01-01", "2024-01-02"])
        result = trainer.run(X, y, dates, min_train_days=252)
        assert result.oof_predictions is None

    def test_empty_result_on_no_folds(self, Xy_long):
        X, y, dates = Xy_long
        trainer = WalkForwardTrainer()

        def _mock_empty(self, X, y, dates, train_end, purge_days, eval_days):
            return ModelResult()

        with patch.object(LightGBMPredictor, "train_predict", _mock_empty):
            result = trainer.run(X, y, dates, min_train_days=252)
            assert result.oof_predictions is None

    def test_custom_params_passed_to_lgbm(self, Xy_long):
        X, y, dates = Xy_long
        trainer = WalkForwardTrainer(lgbm_params={"num_leaves": 63, "learning_rate": 0.01})

        call_count = [0]

        def _mock_train(self, X, y, dates, train_end, purge_days, eval_days):
            n = call_count[0]
            call_count[0] += 1
            start = 10 * n
            idx = X.index[start:start + 10]
            return ModelResult(
                predictions=pd.Series([0.01] * 10, index=idx),
                feature_importance=pd.Series([1.0], index=["f1"]),
                train_scores={},
            )

        with patch.object(LightGBMPredictor, "train_predict", _mock_train):
            result = trainer.run(X, y, dates, min_train_days=252)

        assert trainer.lgbm_params["num_leaves"] == 63
        assert trainer.lgbm_params["learning_rate"] == 0.01
