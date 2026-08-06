"""自动重训触发 - 基于性能退化和数据漂移的模型自动更新"""
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from smartalpha.config import MODEL_SAVE_DIR as MODEL_DIR, DATA_DIR, PROCESSED_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


class RetrainTrigger:
    """自动重训触发器：监控→判断→执行重训流水线"""

    def __init__(self):
        self.ic_threshold = -0.05       # IC相对下降阈值
        self.drift_threshold = 3        # 漂移特征数阈值
        self.min_retrain_interval = 7   # 最短重训间隔(天)
        self.last_retrain_time = None
        self.trigger_log = []

    def _load_state(self):
        """加载上次重训状态"""
        state_path = MODEL_DIR / "retrain_state.json"
        if state_path.exists():
            try:
                import json
                with open(state_path) as f:
                    state = json.load(f)
                self.last_retrain_time = datetime.fromisoformat(state.get("last_time"))
                self.trigger_log = state.get("log", [])
                return state
            except Exception as e:
                logger.warning(f"状态加载失败: {e}")
        return {}

    def _save_state(self):
        """保存重训状态"""
        import json
        state = {
            "last_time": datetime.now().isoformat(),
            "log": self.trigger_log[-20:],  # 保留最近20条
        }
        with open(MODEL_DIR / "retrain_state.json", "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def check_should_retrain(self, current_ic=None, current_features=None):
        """
        综合判断是否触发重训
        返回: (should_retrain: bool, reasons: list)
        """
        self._load_state()
        reasons = []

        # 1. 时间间隔检查
        if self.last_retrain_time:
            days_since = (datetime.now() - self.last_retrain_time).days
            if days_since < self.min_retrain_interval:
                return False, [f"距离上次重训仅{days_since}天，间隔不足"]

        # 2. 性能退化检查
        if current_ic is not None:
            baseline_ic = self._load_baseline_ic()
            if baseline_ic is not None:
                delta = current_ic - baseline_ic
                if delta < self.ic_threshold:
                    reasons.append(f"IC退化: {baseline_ic:.4f} -> {current_ic:.4f} (delta={delta:.4f})")

        # 3. 数据漂移检查
        if current_features is not None:
            drift_count = self._check_feature_drift(current_features)
            if drift_count >= self.drift_threshold:
                reasons.append(f"数据漂移: {drift_count}个特征分布异常")

        # 4. 模型文件过期检查
        model_age = self._get_model_age_days()
        if model_age and model_age > 30:
            reasons.append(f"模型已过期: {model_age}天未更新")

        should_retrain = len(reasons) > 0
        if should_retrain:
            logger.warning(f"触发重训: {reasons}")
        else:
            logger.info("未触发重训，模型状态正常")
        return should_retrain, reasons

    def _load_baseline_ic(self):
        """加载基准IC"""
        try:
            path = MODEL_DIR / "lgbm_baseline_ic.json"
            if path.exists():
                import json
                with open(path) as f:
                    return json.load(f).get("baseline_ic")
            # 尝试从特征重要性目录找历史记录
            result_path = RESULTS_DIR / "backtest_result.parquet"
            if result_path.exists():
                # 如果没有基准，用当前回测的Sharpe作为代理
                return None
        except Exception as e:
            logger.warning(f"基准IC加载失败: {e}")
        return None

    def _save_baseline_ic(self, ic):
        """保存基准IC"""
        import json
        with open(MODEL_DIR / "lgbm_baseline_ic.json", "w") as f:
            json.dump({"baseline_ic": ic, "updated_at": datetime.now().isoformat()}, f)

    def _check_feature_drift(self, current_features):
        """检查特征漂移数量：使用训练时的快照作baseline，而非全量数据"""
        try:
            # 尝试加载训练时的baseline快照
            baseline_path = MODEL_DIR / "feature_baseline.parquet"
            if baseline_path.exists():
                baseline = pd.read_parquet(baseline_path)
            else:
                # 首次运行时用前半段数据作baseline并保存
                full = pd.read_parquet(PROCESSED_DIR / "factors_neutral.parquet")
                full = full.sort_values("trade_date")
                baseline = full.head(len(full) // 2)
                baseline.to_parquet(baseline_path)
            baseline_stats = baseline.describe()
            current_stats = current_features.describe()

            drift_count = 0
            for col in current_features.columns:
                if col in baseline_stats.columns:
                    mean_delta = abs(current_stats[col]["mean"] - baseline_stats[col]["mean"])
                    base_std = baseline_stats[col]["std"]
                    if base_std > 0 and mean_delta / base_std > 2.0:
                        drift_count += 1
            return drift_count
        except Exception as e:
            logger.warning(f"漂移检查失败: {e}")
            return 0

    def _get_model_age_days(self):
        """获取模型文件年龄"""
        try:
            model_path = MODEL_DIR / "lgbm_model.joblib"
            if model_path.exists():
                mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
                return (datetime.now() - mtime).days
        except Exception:
            pass
        return None

    def execute_retrain_pipeline(self, use_optuna=False, selected_factors=None):
        """
        执行完整重训流水线:
        1. 重新准备数据
        2. (可选)Optuna调参
        3. 训练LightGBM
        4. 训练Transformer
        5. 生成集成预测
        6. 更新基准IC
        """
        logger.info("=" * 50)
        logger.info("启动自动重训流水线")
        logger.info("=" * 50)

        # 1. 数据刷新
        logger.info("[1/5] 刷新数据...")
        # 注: 生产环境通过 main.py 调用 run_data_pipeline/daily_fetch
        # 当前直接加载已有 parquet 数据
        try:
            import pyarrow.parquet as pq
            _ = pq.read_table(PROCESSED_DIR / "factors_neutral.parquet")
        except Exception:
            logger.warning("无法加载因子数据，重训可能失败")

        # 2. 因子重新计算
        logger.info("[2/5] 重新计算因子...")
        # 注: 生产环境通过 main.py run_factor_engine 重新计算因子
        # 当前直接使用已保存的因子数据
        selected = []
        ic_results = {}
        try:
            factors = pd.read_parquet(PROCESSED_DIR / "factors_neutral.parquet")
            factor_cols = [c for c in factors.columns if c not in {"ts_code","trade_date","industry","circ_mv","log_mv","fwd_ret_5d"}]
            selected = factor_cols[:12]
            ic_results = {"status": "loaded_from_cache", "n_factors": len(selected)}
            logger.info(f"从缓存加载 {len(selected)} 个因子")
        except Exception as e:
            logger.warning(f"因子数据加载失败: {e}")

        # 3. 超参数优化(如启用)
        params = None
        if use_optuna:
            logger.info("[3/5] 超参数优化...")
            from smartalpha.model.tuner import AutoTuner
            params = AutoTuner("lgbm").run(selected_factors=selected_factors or selected)

        # 4. 模型训练
        logger.info("[4/5] 训练模型...")
        from smartalpha.model.lgbm import LGBMTrainer
        trainer = LGBMTrainer(params=params)
        result = trainer.run(selected_factors=selected_factors or selected)

        if result:
            # 更新基准IC
            self._save_baseline_ic(result.get("ic", 0))
            logger.info(f"重训完成: RMSE={result['rmse']:.6f}, IC={result['ic']:.4f}")

            # 5. 生成集成预测
            logger.info("[5/5] 生成预测...")
            from smartalpha.model.ensemble import EnsembleModel
            ensemble = EnsembleModel()
            ensemble.load_models()
            ensemble.generate_ensemble_predictions()

            self.last_retrain_time = datetime.now()
            self.trigger_log.append({
                "time": datetime.now().isoformat(),
                "reason": "auto_retrain",
                "result": result,
            })
            self._save_state()
            logger.info("自动重训流水线完成")
            return result
        else:
            logger.error("重训失败")
            return None

    def run_monitor_and_retrain(self):
        """完整监控+重训入口：实时计算current_ic而非读baseline"""
        logger.info("运行监控检查...")

        # 实时计算当前IC（在最近OOS数据上预测）
        current_ic = self._compute_current_ic()

        # 加载最近特征（按日期排序后取最新）
        current_features = None
        try:
            fac = pd.read_parquet(PROCESSED_DIR / "factors_neutral.parquet")
            fac = fac.sort_values("trade_date")
            feature_cols = [c for c in fac.columns if c not in ["ts_code", "trade_date", "industry", "circ_mv", "log_mv"]]
            current_features = fac[feature_cols].tail(1000)
        except Exception:
            pass

        should_retrain, reasons = self.check_should_retrain(current_ic, current_features)
        if should_retrain:
            return self.execute_retrain_pipeline(use_optuna=True)
        return {"status": "no_retrain_needed", "reasons": reasons}

    def _compute_current_ic(self):
        """在最近OOS数据上实时计算IC，而非读baseline文件"""
        try:
            from smartalpha.model.lgbm import LGBMTrainer
            import joblib
            model_path = MODEL_DIR / "lgbm_model.joblib"
            if not model_path.exists():
                return None
            model = joblib.load(model_path)

            # 加载最新因子数据
            factors = pd.read_parquet(PROCESSED_DIR / "factors_neutral.parquet")
            factors["trade_date"] = pd.to_datetime(factors["trade_date"], format="%Y%m%d")
            daily = pd.read_parquet(PROCESSED_DIR / "daily_masked.parquet")
            daily["trade_date"] = pd.to_datetime(daily["trade_date"], format="%Y%m%d")

            # 计算前瞻收益
            daily = daily.sort_values(["ts_code", "trade_date"])
            daily["fwd_ret_5d"] = daily.groupby("ts_code")["close"].pct_change(5).shift(-5)
            merged = factors.merge(daily[["ts_code", "trade_date", "fwd_ret_5d"]],
                                   on=["ts_code", "trade_date"], how="inner")
            merged = merged.dropna(subset=["fwd_ret_5d"])

            # 取最近30天作为OOS评估窗口
            latest_dates = sorted(merged["trade_date"].unique())[-30:]
            oos = merged[merged["trade_date"].isin(latest_dates)]
            if len(oos) < 100:
                return None

            exclude_cols = {"ts_code", "trade_date", "fwd_ret_5d", "industry", "circ_mv", "log_mv"}
            feature_cols = [c for c in oos.columns if c not in exclude_cols]
            y_pred = model.predict(oos[feature_cols])
            eval_df = pd.DataFrame({"pred": y_pred, "y": oos["fwd_ret_5d"].values, "date": oos["trade_date"].values})
            ic_series = eval_df.groupby("date").apply(lambda g: g["pred"].corr(g["y"]) if len(g) > 2 else np.nan)
            current_ic = ic_series.mean()
            logger.info(f"当前OOS截面IC: {current_ic:.4f}")
            return current_ic
        except Exception as e:
            logger.warning(f"计算current_ic失败: {e}")
            return None


def trigger_retrain_if_needed():
    """便捷函数：供定时任务或监控脚本调用"""
    trigger = RetrainTrigger()
    return trigger.run_monitor_and_retrain()
