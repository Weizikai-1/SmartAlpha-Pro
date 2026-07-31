"""模型集成 - LightGBM + Transformer Stacking/Blending"""
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

# 模型存储路径 (兼容旧 config.MODEL_DIR)
MODEL_DIR = Path(__file__).parent / "saved"
MODEL_DIR.mkdir(exist_ok=True)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class EnsembleModel:
    """模型集成: LightGBM + Transformer stacking"""

    def __init__(self, lgbm_weight=0.7, transformer_weight=0.3):
        self.lgbm_weight = lgbm_weight
        self.transformer_weight = transformer_weight
        self.lgbm_model = None
        self.transformer_model = None
        self.meta_model = None

    def load_models(self):
        """加载已训练的模型"""
        # 加载LightGBM
        try:
            lgbm_path = MODEL_DIR / "lgbm_model.joblib"
            self.lgbm_model = joblib.load(lgbm_path)
            logger.info(f"LightGBM模型加载: {lgbm_path}")
        except FileNotFoundError:
            logger.error("LightGBM模型不存在")

        # 加载Transformer（从保存的超参数重建模型）
        if HAS_TORCH:
            try:
                from smartalpha.model.transformer import TransformerModel
                import json
                trans_path = MODEL_DIR / "transformer_model.pt"
                hp_path = MODEL_DIR / "transformer_hyperparams.json"
                input_dim = 64
                if hp_path.exists():
                    with open(hp_path) as f:
                        hp = json.load(f)
                    input_dim = hp.get("input_dim", 64)
                self.transformer_model = TransformerModel(input_dim=input_dim)
                self.transformer_model.load_state_dict(torch.load(trans_path))
                self.transformer_model.eval()
                logger.info(f"Transformer模型加载: {trans_path} (input_dim={input_dim})")
            except FileNotFoundError:
                logger.warning("Transformer模型不存在")

    def predict_blend(self, X, ts_codes=None):
        """加权融合：LightGBM直接预测，Transformer按股票分组构建3D序列"""
        predictions = []
        weights = []

        if self.lgbm_model is not None:
            pred_lgbm = self.lgbm_model.predict(X)
            predictions.append(pred_lgbm)
            weights.append(self.lgbm_weight)

        if self.transformer_model is not None and HAS_TORCH:
            from smartalpha.model.transformer import TransformerTrainer
            trainer = TransformerTrainer(input_dim=X.shape[1])
            pred_trans = trainer.predict(X, ts_codes=ts_codes)
            if pred_trans is not None and len(pred_trans) > 0:
                # Transformer输出长度可能与LGBM不同，取最后N个对齐
                n = len(pred_lgbm) if 'pred_lgbm' in dir() else len(X)
                if len(pred_trans) >= n:
                    pred_trans = pred_trans[-n:]
                else:
                    # 不足部分用LGBM填充
                    if 'pred_lgbm' in dir():
                        pred_trans = np.concatenate([pred_lgbm[:n-len(pred_trans)], pred_trans])
                predictions.append(pred_trans)
                weights.append(self.transformer_weight)

        if not predictions:
            logger.error("没有可用模型")
            return None

        weights = np.array(weights) / sum(weights)
        blended = sum(w * p for w, p in zip(weights, predictions))
        return blended

    def train_stacking(self, X, y, dates, test_ratio=0.3):
        """Stacking集成训练"""
        from smartalpha.model.lgbm import LGBMTrainer
        from smartalpha.model.transformer import TransformerTrainer

        unique_dates = sorted(dates.unique())
        # 【P0修复】添加边界检查，确保split_idx不超过数组长度
        split_idx = int(len(unique_dates) * (1 - test_ratio))
        # 确保至少有一个训练日期
        split_idx = max(1, min(split_idx, len(unique_dates) - 1))
        split_date = unique_dates[split_idx]

        train_mask = dates < split_date
        X_train, y_train = X[train_mask], y[train_mask]

        # 第一层: 训练基学习器
        logger.info("第一层: 训练LightGBM...")
        lgbm_trainer = LGBMTrainer()
        # 【P0修复】修改test_ratio为0.2，避免内部索引越界
        lgbm_trainer.train(X_train, y_train, dates[train_mask], test_ratio=0.2)
        self.lgbm_model = lgbm_trainer.model

        if HAS_TORCH:
            logger.info("第一层: 训练Transformer...")
            trans_trainer = TransformerTrainer(input_dim=X_train.shape[1])
            trans_trainer.train(X_train, y_train)
            self.transformer_model = trans_trainer.model

        logger.info("Stacking模型训练完成")
        return self

    def generate_ensemble_predictions(self, factor_file="factors_neutral.parquet"):
        """生成集成预测评分：优先使用blend，回退到单模型"""
        try:
            factors = pd.read_parquet(MODEL_DIR.parent.parent.parent / "data" / "processed" / factor_file)
        except FileNotFoundError:
            logger.error("缺少因子数据")
            return None

        # 排除标签列和元数据列
        exclude_cols = {"ts_code", "trade_date", "industry", "circ_mv", "log_mv", "fwd_ret_5d"}
        feature_cols = [c for c in factors.columns if c not in exclude_cols]
        X = factors[feature_cols]

        if self.lgbm_model is None:
            self.load_models()

        if self.lgbm_model is not None and self.transformer_model is not None:
            # 双模型blend
            ts_codes = factors["ts_code"].values if "ts_code" in factors.columns else None
            factors["predict_score"] = self.predict_blend(X, ts_codes=ts_codes)
        elif self.lgbm_model is not None:
            # 仅LightGBM
            factors["predict_score"] = self.lgbm_model.predict(X)
        else:
            logger.error("无可用模型")
            return None

        result = factors[["ts_code", "trade_date", "predict_score"]].copy()
        result.to_parquet(MODEL_DIR.parent.parent.parent / "data" / "processed" / "model_predictions.parquet")
        logger.info(f"集成预测评分已保存: {len(result)} 条")
        return result