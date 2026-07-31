"""smartalpha.model — ML预测模型层。

提供：
- LightGBM 预测器（时序交叉验证 + Purge防泄漏）
- Walk-Forward 滚动训练管道
- 特征重要性分析
"""

try:
    from .lgbm import LightGBMPredictor
    from .trainer import WalkForwardTrainer
    __all__ = ["LightGBMPredictor", "WalkForwardTrainer"]
except ImportError:
    __all__ = []
    import warnings
    warnings.warn("lightgbm 未安装，model 模块不可用。运行: pip install lightgbm")
