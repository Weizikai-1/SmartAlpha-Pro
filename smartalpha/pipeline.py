"""生产级回测管道 — OOF预测 → 截面回测 一站式对接。

解决问题: WalkForwardTrainer 生成的 OOF 预测如何自动输入 BacktestEngine。
此前 demo.py 用因子排名选股（非 ML 预测），本模块修正为严格的 OOF 预测选股。

管道流程:
1. 输入面板数据 (MultiIndex: date × stock) + 因子值
2. WalkForwardTrainer 滚动训练 → OOF 预测 (样本级)
3. OOF 预测还原为 (date, stock) MultiIndex 信号
4. BacktestEngine 用 OOF 信号 + 风控执行回测
5. 输出: 回测绩效 + OOF 评估指标

生产级数据需求 (诚实文档):
- panel: 需真实日线 OHLCV 数据 (DataLoader 已支持)
- factors: 需通过表达式引擎计算的因子值 (需 MultiIndex 格式)
- forward_returns: 标签需从真实价格计算，且须与 OOF 区间严格对齐
- 模型训练依赖 LightGBM (本机 DLL 可能不可用，参考 test_model_mock.py)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from smartalpha._constants import EPS
from smartalpha.model.trainer import WalkForwardTrainer, WalkForwardResult
from smartalpha.backtest.engine import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


def build_signal_from_predictions(
    oof_predictions: pd.Series,
    panel: pd.DataFrame,
    factor_idx: pd.MultiIndex,
) -> pd.Series:
    """将 OOF 预测还原为截面信号 (date, stock) MultiIndex。

    OOF 预测的 index 是原始样本行号，需要映射回 (date, stock)。

    Args:
        oof_predictions: WalkForwardResult.oof_predictions。
        panel: 原始面板数据 (MultiIndex)。
        factor_idx: 因子值对应的 MultiIndex (用于对齐)。

    Returns:
        signal Series (MultiIndex: trade_date × ts_code)，索引与 panel 对齐。
        NaN 表示该日无预测。

    数据要求 (生产级):
        - oof_predictions.index 必须与 factor_idx 的某个维度对应。
        - 如果因子是截面因子 (每个 date × stock 一行)，则 OOF 预测可直接映射。
    """
    if oof_predictions is None or oof_predictions.empty:
        return pd.Series(dtype=float)

    # 尝试从 OOF 预测的 index 映射到 factor_idx
    common = oof_predictions.index.intersection(factor_idx)
    if len(common) > 0:
        # OOF index 直接是 MultiIndex → 直接使用
        signal = pd.Series(np.nan, index=factor_idx, dtype=float)
        signal.loc[common] = oof_predictions.loc[common]
        return signal

    # 备选: OOF index 是整数/默认索引，尝试按位置对齐
    # (仅当 factor_idx 与 panel 的长度一致时)
    if len(oof_predictions) == len(factor_idx):
        signal = pd.Series(oof_predictions.values, index=factor_idx, dtype=float)
        return signal

    # 无法对齐 → 空信号
    return pd.Series(dtype=float)


class CrossSectionalPipeline:
    """截面回测管道: 因子 → ML训练 → OOF预测 → 回测。

    使用示例:
        pipe = CrossSectionalPipeline(
            trainer=WalkForwardTrainer(purge_days=5, val_days=60, step_days=60),
            engine=BacktestEngine(top_n=20, rebalance_freq="M"),
            risk_manager=RiskManager(),
        )
        result = pipe.run(panel, factor_df, forward_returns, industry_map)

    数据要求 (生产级):
        - panel: MultiIndex(date, stock), 含 "close" 列
        - factor_df: date×stock 因子宽表 或 MultiIndex 因子值
        - forward_returns: 与 factor_df 对齐的前向收益率 Series
        - industry_map: dict[stock→industry], 风控时必传
    """

    def __init__(
        self,
        trainer: Optional[WalkForwardTrainer] = None,
        engine: Optional[BacktestEngine] = None,
        risk_manager: Optional[object] = None,
    ) -> None:
        """初始化管道。

        Args:
            trainer: WalkForward 训练器，None 则使用默认参数。
            engine: 回测引擎，None 则使用默认 Top20/月度调仓。
            risk_manager: 风控管理器，None 则不启用。
        """
        self.trainer = trainer or WalkForwardTrainer()
        self.engine = engine or BacktestEngine(top_n=20, rebalance_freq="M")
        self.risk_manager = risk_manager

    def run(
        self,
        panel: pd.DataFrame,
        factor_df: pd.DataFrame | pd.Series,
        forward_returns: pd.Series,
        industry_map: Optional[dict] = None,
        price_col: str = "close",
    ) -> dict:
        """执行完整管道: 训练 → OOF预测 → 回测。

        Args:
            panel: 面板数据 (MultiIndex: date × stock)。
            factor_df: 因子数据。
                       - 宽表: date × stock，每列一个因子。
                       - 或 MultiIndex Series (date, stock) 一个因子。
            forward_returns: 前向收益率 (index 需与 factor_df 对齐)。
            industry_map: 股票→行业 映射 (风控时必传)。
            price_col: 价格列名。

        Returns:
            {
                "train_result": WalkForwardResult (含 OOF 预测和 IC/RMSE),
                "backtest_result": BacktestResult (含净值曲线和绩效指标),
                "signal": 用于回测的信号序列,
            }
        """
        # Step 1: 准备训练数据
        if isinstance(factor_df.index, pd.MultiIndex):
            # 单个因子 → 转宽表
            factor_wide = factor_df.unstack("ts_code")
        else:
            factor_wide = factor_df

        # 获取因子 MultiIndex 用于 OOF 映射
        if isinstance(factor_df.index, pd.MultiIndex):
            factor_idx = factor_df.index
        else:
            # 非 MultiIndex 时创建空索引
            factor_idx = pd.MultiIndex.from_arrays(
                [[], []], names=["trade_date", "ts_code"]
            )

        # Step 2: Walk-Forward 训练
        dates_series = pd.Series(panel.index.get_level_values("trade_date").unique())

        # 为 WalkForwardTrainer 准备扁平化特征 + 标签
        # 取所有股票的特征均值作为日度特征 (简化: 生产应逐股票训练)
        # 更实用的做法: 直接传入宽表，每行=日，每列=股票
        # WalkForwardTrainer 需要 (n_samples, n_features) 的 X
        # 这里我们使用截面因子均值作为特征矩阵
        n_dates = len(factor_wide)
        n_factors = factor_wide.shape[1] if len(factor_wide.shape) > 1 else 1

        if n_dates < 252:
            logger.warning(f"数据不足: 仅 {n_dates} 个交易日 (需 ≥252), 跳过管道")
            return {
                "train_result": WalkForwardResult(),
                "backtest_result": BacktestResult(),
                "signal": pd.Series(dtype=float),
                "error": f"数据不足: 仅 {n_dates} 个交易日",
            }

        # 扁平化: 每行 = 一日期 × 一股票
        if isinstance(factor_df.index, pd.MultiIndex):
            X_flat = factor_wide.stack().to_frame("factor_value")
            # 简化: 使用单因子 + 全截面训练
            X_flat = factor_wide
            y_flat = forward_returns.reindex(factor_wide.index)
            dates_for_train = pd.Series(factor_wide.index)
        else:
            X_flat = factor_wide
            y_flat = forward_returns
            dates_for_train = pd.Series(factor_wide.index)

        # Step 3: 训练 OOF
        try:
            train_result = self.trainer.run(X_flat, y_flat, dates_for_train)
        except Exception as e:
            logger.error(f"WalkForward 训练失败: {e}", exc_info=True)
            return {
                "train_result": WalkForwardResult(),
                "backtest_result": BacktestResult(),
                "signal": pd.Series(dtype=float),
                "error": f"训练失败: {e}",
            }

        if train_result.oof_predictions is None:
            logger.warning("OOF 预测为空 (训练集不足或模型未收敛)")
            return {
                "train_result": train_result,
                "backtest_result": BacktestResult(),
                "signal": pd.Series(dtype=float),
            }

        # Step 4: OOF → 信号
        signal = build_signal_from_predictions(
            train_result.oof_predictions, panel, factor_idx
        )

        if signal.empty:
            logger.warning("信号生成失败: build_signal_from_predictions 返回空 Series (OOF index 与 panel 未对齐)")
            return {
                "train_result": train_result,
                "backtest_result": BacktestResult(),
                "signal": signal,
            }

        # Step 5: 回测 (带风控)
        bt_result = self.engine.run(
            panel,
            signal,
            price_col=price_col,
            risk_manager=self.risk_manager,
            industry_map=industry_map,
        )

        return {
            "train_result": train_result,
            "backtest_result": bt_result,
            "signal": signal,
        }
