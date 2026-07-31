"""金融函数库模块。

提供 50+ 因子计算函数，涵盖：
- 基础统计（MEAN, STD, VAR, MAX, MIN, SUM, COUNT, MEDIAN, PRODUCT）
- 排名与标准化（RANK, ZSCORE, SCALE, DENSE_RANK, PERCENTILE）
- 时序操作（DELTA, DELAY, DIFF, HHV, LLV）
- 技术指标（MA, EMA, RSI, MACD, BOLL, KDJ, ATR）
- 相关性（CORR, COVARIANCE, BETA）
- 数学运算（ABS, SIGN, LOG, SQRT, POWER, EXP）
- 截面操作（CS_RANK, CS_ZSCORE, CS_SCALE）
- 条件与筛选（IF, WHEN, FILTER）

所有函数均基于 numpy 实现向量化运算。
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from ._func_helpers import (
    ArrayLike,
    dense_rank_array,
    ema_internal,
    rank_array,
    rolling_apply,
    rolling_apply_two,
    scale_fn,
    shift_array,
    to_array,
)


class FinancialFunctionLibrary:
    """金融函数库。

    内置 50+ 因子函数，支持向量化计算，可通过 ``register`` 扩展自定义函数。

    使用示例::

        lib = FinancialFunctionLibrary()
        result = lib.call("RANK", np.array([3.0, 1.0, 2.0]), 2)
        print(lib.list_functions())
    """

    def __init__(self) -> None:
        self._functions: Dict[str, Callable[..., ArrayLike]] = {}
        self._register_builtins()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def has_function(self, name: str) -> bool:
        """检查函数是否已注册。

        Args:
            name: 函数名（不区分大小写）。

        Returns:
            是否存在。
        """
        return name.upper() in self._functions

    def call(self, name: str, *args: ArrayLike) -> ArrayLike:
        """调用指定函数。

        Args:
            name: 函数名。
            *args: 函数参数。

        Returns:
            计算结果。

        Raises:
            KeyError: 函数未注册时。
        """
        key = name.upper()
        if key not in self._functions:
            raise KeyError(f"函数未注册: {name}")
        return self._functions[key](*args)

    def register(
        self, name: str, func: Callable[..., ArrayLike]
    ) -> None:
        """注册自定义函数。

        Args:
            name: 函数名。
            func: 可调用对象。
        """
        self._functions[name.upper()] = func

    def list_functions(self) -> list[str]:
        """列出所有已注册的函数名。

        Returns:
            函数名列表（按字母排序）。
        """
        return sorted(self._functions.keys())

    # ------------------------------------------------------------------
    # 内置函数注册
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        """注册所有内置函数。"""

        # —— 基础统计 ——
        self._register_statistics()
        # —— 排名与标准化 ——
        self._register_ranking()
        # —— 时序操作 ——
        self._register_timeseries()
        # —— 技术指标 ——
        self._register_indicators()
        # —— 相关性 ——
        self._register_correlation()
        # —— 数学运算 ——
        self._register_math()
        # —— 截面操作 ——
        self._register_cross_section()
        # —— 条件 / 筛选 ——
        self._register_conditional()
        # —— 财务比率 ——
        self._register_financial()

    # ------------------------------------------------------------------
    # 基础统计函数
    # ------------------------------------------------------------------

    def _register_statistics(self) -> None:
        F = self._functions

        def _mean(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动均值；window=0 时为全局均值。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanmean(arr), dtype=np.float64)
            return rolling_apply(arr, window, np.nanmean)

        def _std(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动标准差。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanstd(arr, ddof=1), dtype=np.float64)
            return rolling_apply(arr, window, lambda a: np.nanstd(a, ddof=1))

        def _var(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动方差。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanvar(arr, ddof=1), dtype=np.float64)
            return rolling_apply(arr, window, lambda a: np.nanvar(a, ddof=1))

        def _max(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动最大值。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanmax(arr), dtype=np.float64)
            return rolling_apply(arr, window, np.nanmax)

        def _min(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动最小值。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanmin(arr), dtype=np.float64)
            return rolling_apply(arr, window, np.nanmin)

        def _sum(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动求和。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nansum(arr), dtype=np.float64)
            return rolling_apply(arr, window, np.nansum)

        def _count(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动非空计数。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.sum(~np.isnan(arr)), dtype=np.float64)
            return rolling_apply(arr, window, lambda a: np.sum(~np.isnan(a)))

        def _median(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动中位数。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanmedian(arr), dtype=np.float64)
            return rolling_apply(arr, window, np.nanmedian)

        def _product(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动乘积。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanprod(arr), dtype=np.float64)
            return rolling_apply(arr, window, np.nanprod)

        F["MEAN"] = _mean
        F["STD"] = _std
        F["VAR"] = _var
        F["MAX"] = _max
        F["MIN"] = _min
        F["SUM"] = _sum
        F["COUNT"] = _count
        F["MEDIAN"] = _median
        F["PRODUCT"] = _product

    # ------------------------------------------------------------------
    # 排名与标准化
    # ------------------------------------------------------------------

    def _register_ranking(self) -> None:
        F = self._functions

        def _rank(x: ArrayLike, window: int = 0, ascending: int = 1) -> np.ndarray:
            """滚动排名（百分位）。

            Args:
                x: 输入序列。
                window: 滚动窗口，0 为全局。
                ascending: 1=升序排名，0=降序排名。
            """
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                order = np.argsort(arr) if ascending else np.argsort(-arr)
                ranks = np.empty_like(order, dtype=np.float64)
                ranks[order] = np.arange(1, len(arr) + 1, dtype=np.float64)
                return ranks / len(arr)

            def _rank_last(segment: np.ndarray) -> float:
                ranks = rank_array(segment, ascending)
                return float(ranks[-1])

            return rolling_apply(arr, window, _rank_last)

        def _zscore(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动 Z-Score 标准化。"""
            arr = to_array(x)
            mean = F["MEAN"](arr, window)
            std = F["STD"](arr, window)
            result = (arr - mean) / std
            return np.where(np.isnan(result) | np.isinf(result), 0.0, result)

        def _scale(x: ArrayLike, a: float = 1.0, b: float = 0.0) -> np.ndarray:
            """线性缩放到 [b, b+a]。"""
            arr = to_array(x)
            arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
            rng = arr_max - arr_min
            if rng == 0:
                return np.full_like(arr, b + a / 2, dtype=np.float64)
            return a * (arr - arr_min) / rng + b

        def _dense_rank(x: ArrayLike, window: int = 0) -> np.ndarray:
            """密集排名（相同值并列）。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                unique_vals = np.unique(arr)
                rank_map = {v: i + 1 for i, v in enumerate(unique_vals)}
                return np.array([rank_map[v] for v in arr], dtype=np.float64)

            def _dense_last(segment: np.ndarray) -> float:
                ranks = dense_rank_array(segment)
                return float(ranks[-1])

            return rolling_apply(arr, window, _dense_last)

        def _percentile(x: ArrayLike, pct: float, window: int = 0) -> np.ndarray:
            """滚动分位数。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanpercentile(arr, pct), dtype=np.float64)
            return rolling_apply(arr, window, lambda a: np.nanpercentile(a, pct))

        F["RANK"] = _rank
        F["ZSCORE"] = _zscore
        F["SCALE"] = _scale
        F["DENSE_RANK"] = _dense_rank
        F["PERCENTILE"] = _percentile

    # ------------------------------------------------------------------
    # 时序操作
    # ------------------------------------------------------------------

    def _register_timeseries(self) -> None:
        F = self._functions

        def _delta(x: ArrayLike, period: int = 1) -> np.ndarray:
            """变化率 (x - delay(x, period)) / delay(x, period)。"""
            arr = to_array(x)
            lagged = shift_array(arr, period)
            result = (arr - lagged) / lagged
            return np.where(np.isinf(result), 0.0, result)

        def _delay(x: ArrayLike, period: int = 1) -> np.ndarray:
            """滞后 period 期。"""
            arr = to_array(x)
            return shift_array(arr, period)

        def _diff(x: ArrayLike, period: int = 1) -> np.ndarray:
            """差分 x - delay(x, period)。"""
            arr = to_array(x)
            return arr - shift_array(arr, period)

        def _hhv(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动最高值。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanmax(arr), dtype=np.float64)
            return rolling_apply(arr, window, np.nanmax)

        def _llv(x: ArrayLike, window: int = 0) -> np.ndarray:
            """滚动最低值。"""
            arr = to_array(x)
            if window <= 0 or window >= len(arr):
                return np.full_like(arr, np.nanmin(arr), dtype=np.float64)
            return rolling_apply(arr, window, np.nanmin)

        def _forward(x: ArrayLike) -> np.ndarray:
            """前向填充（用过去值填充 NaN）。"""
            arr = to_array(x)
            result = arr.copy()
            for i in range(1, len(result)):
                if np.isnan(result[i]):
                    result[i] = result[i - 1]
            return result

        def _backward(x: ArrayLike) -> np.ndarray:
            """后向填充（用未来值填充 NaN）。

            ⚠️ 警告：此函数使用未来数据，在回测中使用会导致前向数据泄漏（Look-Ahead Bias）。
            仅应在因子评估（计算IC等需要未来收益的场景）中调用。
            """
            arr = to_array(x)
            result = arr.copy()
            for i in range(len(result) - 2, -1, -1):
                if np.isnan(result[i]):
                    result[i] = result[i + 1]
            return result

        F["DELTA"] = _delta
        F["DELAY"] = _delay
        F["DIFF"] = _diff
        F["HHV"] = _hhv
        F["LLV"] = _llv
        F["FORWARD"] = _forward
        F["BACKWARD"] = _backward

    # ------------------------------------------------------------------
    # 技术指标
    # ------------------------------------------------------------------

    def _register_indicators(self) -> None:
        F = self._functions

        def _ma(x: ArrayLike, window: int = 5) -> np.ndarray:
            """简单移动平均。"""
            arr = to_array(x)
            return rolling_apply(arr, window, np.nanmean)

        def _ema(x: ArrayLike, window: int = 12) -> np.ndarray:
            """指数移动平均。"""
            arr = to_array(x)
            return ema_internal(arr, window)

        def _wma(x: ArrayLike, window: int = 5) -> np.ndarray:
            """加权移动平均（线性权重）。"""
            arr = to_array(x)
            weights = np.arange(1, window + 1, dtype=np.float64)
            weights /= weights.sum()

            result = np.full_like(arr, np.nan)
            for i in range(window - 1, len(arr)):
                segment = arr[i - window + 1 : i + 1]
                result[i] = np.nansum(segment * weights)
            return result

        def _rsi(x: ArrayLike, window: int = 14) -> np.ndarray:
            """相对强弱指标。"""
            arr = to_array(x)
            delta = np.diff(arr, prepend=arr[0])
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)

            avg_gain = np.zeros_like(arr)
            avg_loss = np.zeros_like(arr)
            avg_gain[window - 1] = np.mean(gain[:window])
            avg_loss[window - 1] = np.mean(loss[:window])

            for i in range(window, len(arr)):
                avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
                avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window

            with np.errstate(divide="ignore", invalid="ignore"):
                rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
            return 100.0 - (100.0 / (1.0 + rs))

        def _macd(
            x: ArrayLike,
            fast: int = 12,
            slow: int = 26,
            signal: int = 9,
        ) -> np.ndarray:
            """MACD 柱状图值。"""
            arr = to_array(x)
            ema_fast = ema_internal(arr, fast)
            ema_slow = ema_internal(arr, slow)
            dif = ema_fast - ema_slow
            return ema_internal(dif, signal)

        def _boll(
            x: ArrayLike, window: int = 20, num_std: float = 2.0
        ) -> np.ndarray:
            """布林带中轨。"""
            arr = to_array(x)
            mid = rolling_apply(arr, window, np.nanmean)
            std = rolling_apply(arr, window, lambda a: np.nanstd(a, ddof=1))
            return mid + num_std * std

        def _kdj(
            high: ArrayLike,
            low: ArrayLike,
            close: ArrayLike,
            n: int = 9,
            m1: int = 3,
            m2: int = 3,
        ) -> np.ndarray:
            """KDJ 指标中的 K 值。"""
            h = to_array(high)
            l = to_array(low)
            c = to_array(close)

            rsv = np.zeros_like(c)
            for i in range(n - 1, len(c)):
                hh = np.max(h[max(0, i - n + 1) : i + 1])
                ll = np.min(l[max(0, i - n + 1) : i + 1])
                if hh == ll:
                    rsv[i] = 50.0
                else:
                    rsv[i] = (c[i] - ll) / (hh - ll) * 100.0

            k = np.zeros_like(c)
            k[0] = 50.0
            for i in range(1, len(c)):
                k[i] = (m1 - 1) / m1 * k[i - 1] + 1 / m1 * rsv[i]
            return k

        def _atr(
            high: ArrayLike,
            low: ArrayLike,
            close: ArrayLike,
            window: int = 14,
        ) -> np.ndarray:
            """平均真实波幅。"""
            h = to_array(high)
            l = to_array(low)
            c = to_array(close)
            prev_close = shift_array(c, 1)
            tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
            return rolling_apply(tr, window, np.nanmean)

        def _natr(
            high: ArrayLike,
            low: ArrayLike,
            close: ArrayLike,
            window: int = 14,
        ) -> np.ndarray:
            """归一化平均真实波幅。"""
            atr_val = _atr(high, low, close, window)
            c = to_array(close)
            return atr_val / c * 100.0

        def _sar(
            high: ArrayLike,
            low: ArrayLike,
            acc_init: float = 0.02,
            acc_max: float = 0.20,
        ) -> np.ndarray:
            """抛物线 SAR 指标。

            公式:
                uptrend:  SAR_t = SAR_{t-1} + AF * (EP - SAR_{t-1})
                downtrend: SAR_t = SAR_{t-1} - AF * (SAR_{t-1} - EP)
                AF 每次创极值递增 acc_init，上限 acc_max。

            Args:
                high: 最高价序列。
                low: 最低价序列。
                acc_init: 加速因子初始值，默认 0.02。
                acc_max: 加速因子上限，默认 0.20。
            """
            h = to_array(high)
            l = to_array(low)
            n = len(h)
            sar = np.full(n, np.nan)

            # 初始趋势由前两根K线决定
            if n < 2:
                return sar

            init_up = h[1] >= h[0]
            af = acc_init
            ep = np.nanmax(h[:2]) if init_up else np.nanmin(l[:2])
            sar_val = np.nanmin(l[:2]) if init_up else np.nanmax(h[:2])
            is_uptrend = init_up
            sar[0] = sar_val

            for i in range(1, n):
                prev_sar = sar_val
                sar_val = prev_sar + af * (ep - prev_sar)

                if is_uptrend:
                    # 上趋势中 SAR 不能高于前两根K线最低点
                    floor = min(l[max(0, i - 1)], l[i])
                    sar_val = min(sar_val, floor)
                    # 趋势反转检查: 价格跌破 SAR
                    if l[i] < sar_val:
                        is_uptrend = False
                        sar_val = ep
                        ep = l[i]
                        af = acc_init
                    else:
                        if h[i] > ep:
                            ep = h[i]
                            af = min(af + acc_init, acc_max)
                else:
                    # 下趋势中 SAR 不能低于前两根K线最高点
                    ceiling = max(h[max(0, i - 1)], h[i])
                    sar_val = max(sar_val, ceiling)
                    # 趋势反转检查: 价格突破 SAR
                    if h[i] > sar_val:
                        is_uptrend = True
                        sar_val = ep
                        ep = h[i]
                        af = acc_init
                    else:
                        if l[i] < ep:
                            ep = l[i]
                            af = min(af + acc_init, acc_max)
                sar[i] = sar_val

            return sar

        def _obv(
            close: ArrayLike, volume: ArrayLike,
        ) -> np.ndarray:
            """能量潮 OBV 指标。

            公式:
                OBV_t = OBV_{t-1} + volume_t  (若 close_t > close_{t-1})
                OBV_t = OBV_{t-1} - volume_t  (若 close_t < close_{t-1})
                OBV_t = OBV_{t-1}             (若 close_t == close_{t-1})

            Args:
                close: 收盘价序列。
                volume: 成交量序列。
            """
            c = to_array(close)
            v = to_array(volume)
            n = len(c)
            obv = np.zeros(n)
            obv[0] = v[0]
            for i in range(1, n):
                if c[i] > c[i - 1]:
                    obv[i] = obv[i - 1] + v[i]
                elif c[i] < c[i - 1]:
                    obv[i] = obv[i - 1] - v[i]
                else:
                    obv[i] = obv[i - 1]
            return obv

        F["MA"] = _ma
        F["EMA"] = _ema
        F["WMA"] = _wma
        F["RSI"] = _rsi
        F["MACD"] = _macd
        F["BOLL"] = _boll
        F["KDJ"] = _kdj
        F["ATR"] = _atr
        F["NATR"] = _natr
        F["SAR"] = _sar
        F["OBV"] = _obv

    # ------------------------------------------------------------------
    # 相关性
    # ------------------------------------------------------------------

    def _register_correlation(self) -> None:
        F = self._functions

        def _corr(x: ArrayLike, y: ArrayLike, window: int = 20) -> np.ndarray:
            """滚动皮尔逊相关系数。"""
            arr_x = to_array(x)
            arr_y = to_array(y)
            return rolling_apply_two(
                arr_x, arr_y, window, lambda a, b: np.corrcoef(a, b)[0, 1]
            )

        def _covariance(x: ArrayLike, y: ArrayLike, window: int = 20) -> np.ndarray:
            """滚动协方差。"""
            arr_x = to_array(x)
            arr_y = to_array(y)
            return rolling_apply_two(
                arr_x, arr_y, window, lambda a, b: np.cov(a, b, ddof=1)[0, 1]
            )

        def _beta(
            x: ArrayLike, y: ArrayLike, window: int = 20
        ) -> np.ndarray:
            """滚动 Beta 系数（x 对 y 的回归斜率）。"""
            arr_x = to_array(x)
            arr_y = to_array(y)

            def _calc_beta(a: np.ndarray, b: np.ndarray) -> float:
                cov = np.cov(a, b, ddof=1)
                var_b = np.var(b, ddof=1)
                if var_b == 0:
                    return 0.0
                return cov[0, 1] / var_b

            return rolling_apply_two(arr_x, arr_y, window, _calc_beta)

        F["CORR"] = _corr
        F["COVARIANCE"] = _covariance
        F["BETA"] = _beta

    # ------------------------------------------------------------------
    # 数学运算
    # ------------------------------------------------------------------

    def _register_math(self) -> None:
        F = self._functions

        def _abs_fn(x: ArrayLike) -> np.ndarray:
            return np.abs(to_array(x))

        def _sign(x: ArrayLike) -> np.ndarray:
            arr = to_array(x)
            return np.sign(arr).astype(np.float64)

        def _log(x: ArrayLike) -> np.ndarray:
            arr = to_array(x)
            return np.log(np.where(arr > 0, arr, 1e-10))

        def _log10(x: ArrayLike) -> np.ndarray:
            arr = to_array(x)
            return np.log10(np.where(arr > 0, arr, 1e-10))

        def _exp(x: ArrayLike) -> np.ndarray:
            return np.exp(to_array(x))

        def _sqrt(x: ArrayLike) -> np.ndarray:
            arr = to_array(x)
            return np.sqrt(np.where(arr >= 0, arr, 0.0))

        def _power(x: ArrayLike, p: float = 2.0) -> np.ndarray:
            return np.power(to_array(x), p)

        def _sign_pow(x: ArrayLike, p: float = 0.5) -> np.ndarray:
            """保留符号的幂运算。"""
            arr = to_array(x)
            return np.sign(arr) * np.power(np.abs(arr), p)

        F["ABS"] = _abs_fn
        F["SIGN"] = _sign
        F["LOG"] = _log
        F["LOG10"] = _log10
        F["EXP"] = _exp
        F["SQRT"] = _sqrt
        F["POWER"] = _power
        F["SIGN_POW"] = _sign_pow

    # ------------------------------------------------------------------
    # 截面操作
    # ------------------------------------------------------------------

    def _register_cross_section(self) -> None:
        F = self._functions

        def _cs_rank(x: ArrayLike) -> np.ndarray:
            """截面排名（每个时间点独立排名）。"""
            arr = to_array(x)
            return rank_array(arr, ascending=True)

        def _cs_zscore(x: ArrayLike) -> np.ndarray:
            """截面 Z-Score。"""
            arr = to_array(x)
            mean = np.nanmean(arr)
            std = np.nanstd(arr, ddof=1)
            result = (arr - mean) / std
            return np.where(np.isnan(result), 0.0, result)

        def _cs_scale(x: ArrayLike, a: float = 1.0, b: float = 0.0) -> np.ndarray:
            """截面缩放。"""
            arr = to_array(x)
            return scale_fn(arr, a, b)

        def _cs_median(x: ArrayLike) -> np.ndarray:
            """截面去均值（中位数）。"""
            arr = to_array(x)
            return arr - np.nanmedian(arr)

        def _cs_abs(x: ArrayLike) -> np.ndarray:
            """截面绝对值。"""
            return np.abs(to_array(x))

        F["CS_RANK"] = _cs_rank
        F["CS_ZSCORE"] = _cs_zscore
        F["CS_SCALE"] = _cs_scale
        F["CS_MEDIAN"] = _cs_median
        F["CS_ABS"] = _cs_abs

    # ------------------------------------------------------------------
    # 条件 / 筛选
    # ------------------------------------------------------------------

    def _register_conditional(self) -> None:
        F = self._functions

        def _when(
            condition: ArrayLike, then_val: ArrayLike, else_val: ArrayLike = 0.0
        ) -> np.ndarray:
            """条件三元运算。"""
            cond = to_array(condition).astype(bool)
            then_v = to_array(then_val)
            else_v = to_array(else_val)
            return np.where(cond, then_v, else_v)

        def _filter(
            x: ArrayLike,
            low: Optional[float] = None,
            high: Optional[float] = None,
        ) -> np.ndarray:
            """区间筛选，超出范围置 NaN。"""
            arr = to_array(x)
            result = arr.copy()
            if low is not None:
                result = np.where(result < low, np.nan, result)
            if high is not None:
                result = np.where(result > high, np.nan, result)
            return result

        def _keep(x: ArrayLike, n: int = 1) -> np.ndarray:
            """保留每 n 个数据点。"""
            arr = to_array(x)
            result = np.full_like(arr, np.nan)
            result[::n] = arr[::n]
            return result

        F["WHEN"] = _when
        F["FILTER"] = _filter
        F["KEEP"] = _keep

    # ------------------------------------------------------------------
    # 财务比率
    # ------------------------------------------------------------------

    def _register_financial(self) -> None:
        F = self._functions

        def _leverage(
            debt: ArrayLike, equity: ArrayLike,
        ) -> np.ndarray:
            """杠杆率 — 负债/权益比。

            公式: LEVERAGE = debt / equity

            用于衡量企业财务杠杆水平。值越大，财务风险越高。

            Args:
                debt: 总负债序列。
                equity: 所有者权益序列。
            """
            d = to_array(debt)
            e = to_array(equity)
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(e != 0, d / e, np.nan)
            return result

        def _leverage_ratio(
            debt: ArrayLike, total_assets: ArrayLike,
        ) -> np.ndarray:
            """资产负债率 — 负债/总资产。

            公式: DEBT_RATIO = debt / total_assets

            衡量企业总资产中有多大比例是通过负债融资的。

            Args:
                debt: 总负债序列。
                total_assets: 总资产序列。
            """
            d = to_array(debt)
            a = to_array(total_assets)
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(a != 0, d / a, np.nan)
            return result

        def _growth(
            x: ArrayLike, period: int = 4,
        ) -> np.ndarray:
            """同比增长率。

            公式: GROWTH = (x_t / x_{t-period}) - 1

            常用于计算营收增长率、净利润增长率等财务指标。
            period=4 表示同比（季度数据），period=1 表示环比。

            Args:
                x: 财务指标序列（如季度营收）。
                period: 滞后期数，默认 4（同比）。
            """
            arr = to_array(x)
            lagged = shift_array(arr, period)
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(lagged != 0, (arr / lagged) - 1, np.nan)
            return result

        def _roic(
            net_profit: ArrayLike, invested_capital: ArrayLike,
        ) -> np.ndarray:
            """投入资本回报率。

            公式: ROIC = net_profit / (total_assets - current_liabilities)

            衡量企业运用投入资本的效率。

            Args:
                net_profit: 净利润序列。
                invested_capital: 投入资本序列（总资产 - 无息流动负债）。
            """
            n = to_array(net_profit)
            ic = to_array(invested_capital)
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(ic != 0, n / ic, np.nan)
            return result

        F["LEVERAGE"] = _leverage
        F["DEBT_RATIO"] = _leverage_ratio
        F["GROWTH"] = _growth
        F["ROIC"] = _roic


