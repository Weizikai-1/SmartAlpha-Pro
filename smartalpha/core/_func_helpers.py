"""函数库内部辅助 — 数组操作、滚动窗口、排名等底层工具。

本模块仅供 functions.py 内部使用，不对外暴露。
"""

from __future__ import annotations

from typing import Callable, List, Union

import numpy as np

Number = Union[float, int]
ArrayLike = Union[Number, List[float], "np.ndarray"]


def to_array(x) -> np.ndarray:
    """将输入统一转为 float64 数组。"""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64)
    if isinstance(x, (list, tuple)):
        return np.array(x, dtype=np.float64)
    if isinstance(x, (int, float)):
        return np.array([float(x)], dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def shift_array(arr: np.ndarray, period: int) -> np.ndarray:
    """滞后 period 期，前 period 个位置填 NaN。"""
    period = max(0, int(period))
    result = np.full_like(arr, np.nan, dtype=np.float64)
    if period == 0:
        result[:] = arr
    else:
        result[period:] = arr[:-period]
    return result


def rolling_apply(
    arr: np.ndarray, window: int, func: Callable[[np.ndarray], float]
) -> np.ndarray:
    """对数组应用滚动窗口聚合。委托 pandas 实现 C 级性能。

    窗口不足 min_periods 或数组为空时返回全 NaN。
    """
    window = max(1, int(window))
    n = len(arr)
    if n == 0 or n < window:
        return np.full(n, np.nan, dtype=np.float64)

    import pandas as pd
    s = pd.Series(arr)
    rolled = s.rolling(window=window, min_periods=window).apply(func, raw=True)
    result = np.full(n, np.nan, dtype=np.float64)
    result[:] = rolled.values
    return result


def rolling_apply_two(
    arr_x: np.ndarray,
    arr_y: np.ndarray,
    window: int,
    func: Callable[[np.ndarray, np.ndarray], float],
) -> np.ndarray:
    """对两个数组应用滚动窗口函数（如 CORR、COV）。

    使用 sliding_window_view 实现，避免 DataFrame.rolling() 逐列调用的 1D/2D 维度问题。
    """
    window = max(1, int(window))
    n = len(arr_x)
    if n == 0 or n < window:
        return np.full(n, np.nan, dtype=np.float64)

    from numpy.lib.stride_tricks import sliding_window_view

    x_windows = sliding_window_view(arr_x, window)
    y_windows = sliding_window_view(arr_y, window)

    result = np.full(n, np.nan, dtype=np.float64)
    for i in range(len(x_windows)):
        xi, yi = x_windows[i], y_windows[i]
        if np.all(np.isnan(xi)) or np.all(np.isnan(yi)):
            result[window - 1 + i] = np.nan
        else:
            result[window - 1 + i] = func(xi, yi)

    return result


def rank_array(arr: np.ndarray, ascending: bool = True) -> np.ndarray:
    """百分位排名。返回 [0, 1] 区间值。"""
    n = len(arr)
    if n == 0:
        return arr.copy()
    order = np.argsort(arr) if ascending else np.argsort(-arr)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    return ranks / n


def dense_rank_array(arr: np.ndarray) -> np.ndarray:
    """密集排名（相同值并列）。"""
    unique_vals = np.unique(arr)
    rank_map = {v: float(i + 1) for i, v in enumerate(unique_vals)}
    return np.array([rank_map[v] for v in arr], dtype=np.float64)


def scale_fn(arr: np.ndarray, a: float = 1.0, b: float = 0.0) -> np.ndarray:
    """线性缩放至 [b, b+a] 区间。"""
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    rng = arr_max - arr_min
    if rng == 0:
        return np.full_like(arr, b + a / 2, dtype=np.float64)
    return a * (arr - arr_min) / rng + b


def ema_internal(arr: np.ndarray, window: int) -> np.ndarray:
    """指数移动平均。使用 pandas EWM 替代手动循环。"""
    if len(arr) == 0:
        return arr.astype(np.float64)
    import pandas as pd
    alpha = 2.0 / (window + 1)
    s = pd.Series(arr)
    return s.ewm(alpha=alpha, adjust=False).mean().values
