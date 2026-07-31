"""金融函数库 FinancialFunctionLibrary 测试套件。

覆盖:
- RANK, ZSCORE, SCALE, DENSE_RANK, PERCENTILE
- DELTA, DELAY, DIFF, HHV, LLV
- MEAN, STD, VAR, MAX, MIN, SUM, COUNT, MEDIAN, PRODUCT
- CORR, COVARIANCE, BETA
- RSI, MA, EMA, MACD, BOLL, KDJ, ATR
- ABS, SIGN, LOG, SQRT, POWER, EXP
- CS_RANK, CS_ZSCORE
- WHEN, FILTER, KEEP
- 边界条件
"""

import numpy as np
import pytest

from smartalpha.core.functions import FinancialFunctionLibrary


class TestBasicStatistics:
    """基础统计函数测试。"""

    def test_mean_global(self, func_lib):
        """全局均值。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("MEAN", arr)
        assert np.allclose(result, 3.0)

    def test_mean_window(self, func_lib):
        """滚动窗口均值。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("MEAN", arr, 3)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert abs(result[2] - 2.0) < 1e-10
        assert abs(result[4] - 4.0) < 1e-10

    def test_std_global(self, func_lib):
        """全局标准差。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("STD", arr)
        expected = np.nanstd(arr, ddof=1)
        assert np.allclose(result, expected)

    def test_std_window(self, func_lib):
        """滚动窗口标准差。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("STD", arr, 3)
        assert len(result) == 5
        assert np.isnan(result[0])

    def test_var_global(self, func_lib):
        """全局方差。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("VAR", arr)
        expected = np.nanvar(arr, ddof=1)
        assert np.allclose(result, expected)

    def test_max_global(self, func_lib):
        """全局最大值。"""
        arr = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        result = func_lib.call("MAX", arr)
        assert np.allclose(result, 5.0)

    def test_max_window(self, func_lib):
        """滚动窗口最大值。"""
        arr = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
        result = func_lib.call("MAX", arr, 3)
        assert abs(result[2] - 4.0) < 1e-10
        assert abs(result[4] - 5.0) < 1e-10

    def test_min_global(self, func_lib):
        """全局最小值。"""
        arr = np.array([3.0, 1.0, 4.0, 2.0, 5.0])
        result = func_lib.call("MIN", arr)
        assert np.allclose(result, 1.0)

    def test_sum_global(self, func_lib):
        """全局求和。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("SUM", arr)
        assert np.allclose(result, 15.0)

    def test_sum_window(self, func_lib):
        """滚动窗口求和。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("SUM", arr, 3)
        assert abs(result[2] - 6.0) < 1e-10
        assert abs(result[4] - 12.0) < 1e-10

    def test_count(self, func_lib):
        """非空计数。"""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = func_lib.call("COUNT", arr)
        assert np.allclose(result, 3.0)

    def test_median(self, func_lib):
        """中位数。"""
        arr = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        result = func_lib.call("MEDIAN", arr)
        assert np.allclose(result, 5.0)

    def test_product(self, func_lib):
        """乘积。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = func_lib.call("PRODUCT", arr)
        assert np.allclose(result, 24.0)


class TestRanking:
    """排名与标准化测试。"""

    def test_rank_global(self, func_lib):
        """全局排名。"""
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = func_lib.call("RANK", arr)
        assert len(result) == 5
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_rank_ascending(self, func_lib):
        """升序排名。"""
        arr = np.array([10.0, 20.0, 30.0])
        result = func_lib.call("RANK", arr)
        assert result[0] < result[1] < result[2]

    def test_rank_descending(self, func_lib):
        """降序排名。"""
        arr = np.array([10.0, 20.0, 30.0])
        result = func_lib.call("RANK", arr, 0, 0)
        assert result[0] > result[1] > result[2]

    def test_zscore_global(self, func_lib):
        """全局 Z-Score。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("ZSCORE", arr)
        assert len(result) == 5
        assert not np.isnan(result).any()

    def test_scale_default(self, func_lib):
        """默认缩放到 [0, 1]。"""
        arr = np.array([10.0, 20.0, 30.0])
        result = func_lib.call("SCALE", arr)
        assert abs(np.nanmin(result)) < 1e-10
        assert abs(np.nanmax(result) - 1.0) < 1e-10

    def test_scale_custom_range(self, func_lib):
        """自定义缩放范围。"""
        arr = np.array([10.0, 20.0, 30.0])
        result = func_lib.call("SCALE", arr, 100.0, 0.0)
        assert abs(np.nanmin(result)) < 1e-10
        assert abs(np.nanmax(result) - 100.0) < 1e-10

    def test_scale_constant_array(self, func_lib):
        """常量数组缩放。"""
        arr = np.array([5.0, 5.0, 5.0])
        result = func_lib.call("SCALE", arr)
        assert np.allclose(result, 0.5)

    def test_dense_rank(self, func_lib):
        """密集排名。"""
        arr = np.array([1.0, 2.0, 2.0, 3.0])
        result = func_lib.call("DENSE_RANK", arr)
        assert result[0] < result[1]
        assert result[1] == result[2]

    def test_percentile(self, func_lib):
        """分位数。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("PERCENTILE", arr, 50)
        assert np.allclose(result, 3.0)


class TestTimeSeries:
    """时序操作测试。"""

    def test_delta(self, func_lib):
        """变化率。"""
        arr = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
        result = func_lib.call("DELTA", arr, 1)
        assert np.isnan(result[0])
        assert abs(result[1] - 0.1) < 1e-10

    def test_delta_period(self, func_lib):
        """多期变化率。"""
        arr = np.arange(100.0, 110.0)
        result = func_lib.call("DELTA", arr, 5)
        assert np.isnan(result[0])
        assert np.isnan(result[4])
        assert abs(result[5] - 0.05) < 1e-10

    def test_delay(self, func_lib):
        """滞后。"""
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = func_lib.call("DELAY", arr, 1)
        assert np.isnan(result[0])
        assert abs(result[1] - 10.0) < 1e-10

    def test_delay_period_zero(self, func_lib):
        """滞后 0 期应返回原数组。"""
        arr = np.array([1.0, 2.0, 3.0])
        result = func_lib.call("DELAY", arr, 0)
        np.testing.assert_array_almost_equal(result, arr)

    def test_diff(self, func_lib):
        """差分。"""
        arr = np.array([10.0, 20.0, 30.0, 25.0, 40.0])
        result = func_lib.call("DIFF", arr, 1)
        assert np.isnan(result[0])
        assert abs(result[1] - 10.0) < 1e-10
        assert abs(result[3] - (-5.0)) < 1e-10

    def test_hhv(self, func_lib):
        """滚动最高值。"""
        arr = np.array([3.0, 5.0, 2.0, 4.0, 1.0])
        result = func_lib.call("HHV", arr, 3)
        assert abs(result[2] - 5.0) < 1e-10
        assert abs(result[4] - 4.0) < 1e-10

    def test_llv(self, func_lib):
        """滚动最低值。"""
        arr = np.array([3.0, 5.0, 2.0, 4.0, 1.0])
        result = func_lib.call("LLV", arr, 3)
        assert abs(result[2] - 2.0) < 1e-10
        assert abs(result[4] - 1.0) < 1e-10


class TestCorrelation:
    """相关性函数测试。"""

    def test_corr(self, func_lib):
        """皮尔逊相关系数。"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = x + np.random.RandomState(42).randn(10) * 0.1
        result = func_lib.call("CORR", x, y, 5)
        assert len(result) == 10
        assert np.isnan(result[0])
        assert not np.isnan(result[-1])

    def test_covariance(self, func_lib):
        """协方差。"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = x + np.random.RandomState(42).randn(10) * 0.5
        result = func_lib.call("COVARIANCE", x, y, 5)
        assert len(result) == 10
        assert not np.isnan(result[-1])

    def test_beta(self, func_lib):
        """Beta 系数。"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        y = x * 2 + np.random.RandomState(42).randn(10) * 0.1
        result = func_lib.call("BETA", x, y, 5)
        assert len(result) == 10
        assert not np.isnan(result[-1])


class TestTechnicalIndicators:
    """技术指标测试。"""

    def test_ma(self, func_lib):
        """简单移动平均。"""
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = func_lib.call("MA", arr, 3)
        assert len(result) == 5
        assert np.isnan(result[0])
        assert not np.isnan(result[-1])

    def test_ema(self, func_lib):
        """指数移动平均。"""
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = func_lib.call("EMA", arr, 3)
        assert len(result) == 5
        assert not np.isnan(result).any()

    def test_rsi(self, func_lib):
        """RSI 指标。"""
        arr = np.array([10.0, 11.0, 10.5, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0,
                        14.5, 16.0, 15.5, 17.0, 16.5, 18.0, 17.5, 19.0, 18.5, 20.0])
        result = func_lib.call("RSI", arr, 14)
        assert len(result) == len(arr)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)
        assert np.all(result[valid] <= 100)

    def test_macd(self, func_lib):
        """MACD 指标。"""
        arr = np.random.RandomState(42).randn(50).cumsum() + 100
        result = func_lib.call("MACD", arr, 12, 26, 9)
        assert len(result) == 50

    def test_boll(self, func_lib):
        """布林带。"""
        arr = np.random.RandomState(42).randn(50).cumsum() + 100
        result = func_lib.call("BOLL", arr, 20, 2.0)
        assert len(result) == 50

    def test_kdj(self, func_lib):
        """KDJ 指标 (K 值)。"""
        n = 30
        high = np.random.RandomState(42).randn(n).cumsum() + 102
        low = np.random.RandomState(43).randn(n).cumsum() + 98
        close = (high + low) / 2
        result = func_lib.call("KDJ", high, low, close, 9)
        assert len(result) == n

    def test_atr(self, func_lib):
        """平均真实波幅。"""
        n = 30
        high = np.random.RandomState(42).randn(n).cumsum() + 102
        low = np.random.RandomState(43).randn(n).cumsum() + 98
        close = (high + low) / 2
        result = func_lib.call("ATR", high, low, close, 14)
        assert len(result) == n


class TestMathFunctions:
    """数学函数测试。"""

    def test_abs(self, func_lib):
        """绝对值。"""
        arr = np.array([-1.0, 2.0, -3.0, 4.0])
        result = func_lib.call("ABS", arr)
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0, 4.0]))

    def test_sign(self, func_lib):
        """符号函数。"""
        arr = np.array([-5.0, 0.0, 3.0])
        result = func_lib.call("SIGN", arr)
        np.testing.assert_array_almost_equal(result, np.array([-1.0, 0.0, 1.0]))

    def test_log(self, func_lib):
        """对数。"""
        arr = np.array([1.0, np.e, np.e**2])
        result = func_lib.call("LOG", arr)
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10
        assert abs(result[2] - 2.0) < 1e-10

    def test_sqrt(self, func_lib):
        """平方根。"""
        arr = np.array([0.0, 1.0, 4.0, 9.0])
        result = func_lib.call("SQRT", arr)
        np.testing.assert_array_almost_equal(result, np.array([0.0, 1.0, 2.0, 3.0]))

    def test_power(self, func_lib):
        """幂函数。"""
        arr = np.array([2.0, 3.0, 4.0])
        result = func_lib.call("POWER", arr, 3)
        np.testing.assert_array_almost_equal(result, np.array([8.0, 27.0, 64.0]))

    def test_exp(self, func_lib):
        """指数函数。"""
        arr = np.array([0.0, 1.0])
        result = func_lib.call("EXP", arr)
        assert abs(result[0] - 1.0) < 1e-10


class TestCrossSection:
    """截面操作测试。"""

    def test_cs_rank(self, func_lib):
        """截面排名。"""
        arr = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        result = func_lib.call("CS_RANK", arr)
        assert len(result) == 5
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_cs_zscore(self, func_lib):
        """截面 Z-Score。"""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = func_lib.call("CS_ZSCORE", arr)
        assert len(result) == 5


class TestConditionalFunctions:
    """条件与筛选函数测试。"""

    def test_when(self, func_lib):
        """条件三元运算。"""
        cond = np.array([True, False, True])
        then_val = np.array([10.0, 20.0, 30.0])
        else_val = np.array([0.0, 0.0, 0.0])
        result = func_lib.call("WHEN", cond, then_val, else_val)
        np.testing.assert_array_almost_equal(result, np.array([10.0, 0.0, 30.0]))

    def test_filter(self, func_lib):
        """区间筛选。"""
        arr = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
        result = func_lib.call("FILTER", arr, 5.0, 15.0)
        assert np.isnan(result[0])
        assert abs(result[1] - 5.0) < 1e-10
        assert abs(result[2] - 10.0) < 1e-10
        assert np.isnan(result[4])


class TestFunctionLibraryRegistry:
    """函数库注册与查询测试。"""

    def test_has_function(self, func_lib):
        """检查函数是否存在。"""
        assert func_lib.has_function("RANK")
        assert func_lib.has_function("MEAN")
        assert not func_lib.has_function("NONEXISTENT")

    def test_call_unknown_raises(self, func_lib):
        """调用未知函数应抛出 KeyError。"""
        with pytest.raises(KeyError):
            func_lib.call("NONEXISTENT_FUNC")

    def test_register_custom(self, func_lib):
        """注册自定义函数。"""
        func_lib.register("CUSTOM_DOUBLE", lambda x: x * 2)
        assert func_lib.has_function("CUSTOM_DOUBLE")
        result = func_lib.call("CUSTOM_DOUBLE", np.array([1.0, 2.0]))
        np.testing.assert_array_almost_equal(result, np.array([2.0, 4.0]))

    def test_list_functions(self, func_lib):
        """列出所有函数。"""
        funcs = func_lib.list_functions()
        assert len(funcs) > 30
        assert "RANK" in funcs
        assert "MEAN" in funcs
        assert "ZSCORE" in funcs

    def test_case_insensitive(self, func_lib):
        """函数名大小写不敏感。"""
        assert func_lib.has_function("rank")
        assert func_lib.has_function("Rank")
        result_upper = func_lib.call("RANK", np.array([1.0, 2.0, 3.0]))
        result_lower = func_lib.call("rank", np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_almost_equal(result_upper, result_lower)


class TestEdgeCases:
    """边界条件测试。"""

    def test_single_value(self, func_lib):
        """单值数组。"""
        arr = np.array([42.0])
        result = func_lib.call("MEAN", arr)
        assert np.allclose(result, 42.0)

    def test_constant_array(self, func_lib):
        """常量数组。"""
        arr = np.array([5.0, 5.0, 5.0, 5.0])
        result = func_lib.call("STD", arr)
        assert np.allclose(result, 0.0)

    def test_nan_handling(self, func_lib):
        """NaN 值处理。"""
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        result = func_lib.call("MEAN", arr)
        assert not np.isnan(result).any()
        assert np.allclose(result, 3.0)

    def test_very_small_window(self, func_lib):
        """窗口为 1。"""
        arr = np.array([10.0, 20.0, 30.0])
        result = func_lib.call("MEAN", arr, 1)
        np.testing.assert_array_almost_equal(result, arr)

    def test_window_larger_than_data(self, func_lib):
        """窗口大于数据长度。"""
        arr = np.array([1.0, 2.0, 3.0])
        result = func_lib.call("MEAN", arr, 10)
        assert not np.isnan(result).any()

    def test_zero_array(self, func_lib):
        """全零数组。"""
        arr = np.array([0.0, 0.0, 0.0])
        result = func_lib.call("STD", arr)
        assert np.allclose(result, 0.0)

    def test_negative_values(self, func_lib):
        """负值数组。"""
        arr = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        result = func_lib.call("MEAN", arr)
        assert np.allclose(result, 0.0)

    def test_very_large_values(self, func_lib):
        """极大值。"""
        arr = np.array([1e10, 2e10, 3e10])
        result = func_lib.call("SUM", arr)
        assert np.allclose(result, 6e10)

    def test_very_small_values(self, func_lib):
        """极小值。"""
        arr = np.array([1e-10, 2e-10, 3e-10])
        result = func_lib.call("MEAN", arr)
        assert np.allclose(result, 2e-10)

    def test_window_zero(self, func_lib):
        """窗口为 0 时使用全局。"""
        arr = np.array([2.0, 4.0, 6.0])
        result = func_lib.call("MEAN", arr, 0)
        assert np.allclose(result, 4.0)

    def test_window_negative(self, func_lib):
        """窗口为负数时使用全局。"""
        arr = np.array([2.0, 4.0, 6.0])
        result = func_lib.call("MEAN", arr, -5)
        assert np.allclose(result, 4.0)

    def test_filter_no_bounds(self, func_lib):
        """无边界筛选。"""
        arr = np.array([1.0, 5.0, 10.0])
        result = func_lib.call("FILTER", arr)
        np.testing.assert_array_almost_equal(result, arr)

    def test_filter_only_lower(self, func_lib):
        """仅下界筛选。"""
        arr = np.array([1.0, 5.0, 10.0])
        result = func_lib.call("FILTER", arr, 3.0)
        assert np.isnan(result[0])
        assert not np.isnan(result[1])
        assert not np.isnan(result[2])

    def test_filter_only_upper(self, func_lib):
        """仅上界筛选。"""
        arr = np.array([1.0, 5.0, 10.0])
        result = func_lib.call("FILTER", arr, None, 7.0)
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])
        assert np.isnan(result[2])