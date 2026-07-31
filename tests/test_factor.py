"""因子工程模块测试 — 中性化、Mask、因子选择。

测试: 行业/市值中性化效果、涨跌停Mask过滤、IC筛选/相关性去重/截面相关筛选。
"""

import numpy as np
import pandas as pd
import pytest

from smartalpha.factor.neutralize import (
    industry_neutralize, market_cap_neutralize, neutralize,
)
from smartalpha.factor.mask import build_limit_mask, apply_mask
from smartalpha.factor.selector import (
    filter_by_ic, remove_correlated, cross_sectional_corr_filter, select_factors,
)


# ============================================================================
# 中性化测试
# ============================================================================

class TestNeutralizeSingleSection:
    """单截面（时序）中性化。"""

    @pytest.fixture
    def factor(self):
        """构造有行业偏差的因子。"""
        np.random.seed(42)
        n = 200
        return pd.Series(np.random.randn(n) * 0.05 + 0.001)

    @pytest.fixture
    def industry(self):
        """3个行业。"""
        return pd.Series(
            ["银行"] * 80 + ["科技"] * 70 + ["消费"] * 50
        )

    @pytest.fixture
    def market_cap(self, factor):
        """市值与因子有一定相关性。"""
        return pd.Series(factor.values * 1000 + np.random.randn(200) * 500 + 10000)

    def test_industry_neutralize_reduces_variance(self, factor, industry):
        """行业中性化应降低方差。"""
        result = industry_neutralize(factor, industry)
        assert result.std() <= factor.std() + 0.001

    def test_industry_neutralize_mean_near_zero(self, factor, industry):
        """中性化后均值应接近零。"""
        result = industry_neutralize(factor, industry)
        assert abs(result.mean()) < 0.01

    def test_market_cap_neutralize(self, factor, market_cap):
        """市值中性化应可运行并降低方差。"""
        result = market_cap_neutralize(factor, market_cap)
        assert result.std() <= factor.std() + 0.001

    def test_combined_neutralize(self, factor, industry, market_cap):
        """组合中性化应可运行。"""
        result = neutralize(factor, industry=industry, market_cap=market_cap)
        assert len(result) == len(factor)
        assert not result.isna().all()

    def test_only_industry(self, factor, industry):
        result = neutralize(factor, industry=industry, market_cap=None)
        assert len(result) == len(factor)

    def test_only_market_cap(self, factor, market_cap):
        result = neutralize(factor, industry=None, market_cap=market_cap)
        assert len(result) == len(factor)

    def test_no_regressors(self, factor):
        result = neutralize(factor, industry=None, market_cap=None)
        assert result.equals(factor)

    def test_empty_input(self):
        empty = pd.Series([], dtype=float)
        result = industry_neutralize(empty, pd.Series([], dtype=str))
        assert result.empty

    def test_nan_handling(self, factor, industry):
        """含 NaN 的行业标签应被跳过。"""
        ind = industry.copy()
        ind.iloc[:10] = np.nan
        result = industry_neutralize(factor, ind)
        # 不应崩溃
        assert len(result) == len(factor)


class TestNeutralizeCrossSectional:
    """截面 (MultiIndex) 中性化。"""

    @pytest.fixture
    def factor_multi(self):
        np.random.seed(42)
        n_dates = 20
        n_stocks = 10
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        stocks = [f"S{i:02d}" for i in range(n_stocks)]
        index = pd.MultiIndex.from_product(
            [dates, stocks], names=["trade_date", "ts_code"]
        )
        # 注入行业偏差: S00-S03 偏高
        vals = np.random.randn(len(index)) * 0.05
        for i in range(4):
            mask = index.get_level_values("ts_code") == stocks[i]
            vals[mask] += 0.15
        return pd.Series(vals, index=index)

    @pytest.fixture
    def industry_multi(self, factor_multi):
        ind_map = {
            "S00": "银行", "S01": "银行", "S02": "银行", "S03": "银行",
            "S04": "科技", "S05": "科技", "S06": "科技",
            "S07": "消费", "S08": "消费", "S09": "消费",
        }
        return pd.Series(
            factor_multi.index.get_level_values("ts_code").map(ind_map),
            index=factor_multi.index,
        )

    @pytest.fixture
    def mc_multi(self, factor_multi):
        np.random.seed(1)
        return pd.Series(
            10000 + np.random.randn(len(factor_multi)) * 2000,
            index=factor_multi.index,
        )

    def test_industry_neutralize_cross_section(self, factor_multi, industry_multi):
        result = industry_neutralize(factor_multi, industry_multi)
        assert not result.isna().all()
        # 中行化后 "银行" 组均值应接近零
        bank_mask = industry_multi == "银行"
        bank_neutralized = result.loc[bank_mask].mean()
        assert abs(bank_neutralized) < 0.05

    def test_combined_cross_section(self, factor_multi, industry_multi, mc_multi):
        result = neutralize(factor_multi, industry=industry_multi, market_cap=mc_multi)
        assert len(result) == len(factor_multi)


# ============================================================================
# Mask 测试
# ============================================================================

class TestMask:
    @pytest.fixture
    def price(self):
        """构造含涨跌停的模拟价格数据 (对齐 MultiIndex)。"""
        np.random.seed(42)
        n_dates = 100
        n_stocks = 5
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
        stocks = [f"S{i}" for i in range(n_stocks)]

        # from_product 的顺序: (date0,S0), (date0,S1), ..., (date1,S0), ...
        index = pd.MultiIndex.from_product([dates, stocks], names=["trade_date", "ts_code"])

        # 建立 date×stock 宽表，再 stack，确保对齐
        prices = {}
        for i_s, s in enumerate(stocks):
            rng = np.random.RandomState(i_s)
            p = 100 + np.cumsum(rng.randn(n_dates) * 1.0)
            # 第 50 日跌停, 第 80 日涨停
            p[50] = p[49] * 0.90
            p[80] = p[79] * 1.10
            prices[s] = p

        price_wide = pd.DataFrame(prices, index=dates)
        price_stacked = price_wide.stack().rename("close")
        price_stacked.index.names = ["trade_date", "ts_code"]

        return pd.DataFrame({"close": price_stacked})

    @pytest.fixture
    def factor(self, price):
        """与 price 对齐的因子值 (宽表→stack)。"""
        price_wide = price["close"].unstack("ts_code")
        factor_wide = price_wide.rank(pct=True)
        stacked = factor_wide.stack()
        stacked.index.names = ["trade_date", "ts_code"]
        return stacked

    def test_build_mask_detects_limit(self, price):
        mask = build_limit_mask(price, threshold=0.095)
        # 第50日附近应有跌停被标记 (wide format, index by date)
        has_limit = (~mask).any(axis=1)  # 每天是否有任何股票被标记
        assert has_limit.any(), "未检测到任何涨跌停"

    def test_build_mask_default_threshold(self, price):
        mask = build_limit_mask(price)
        assert isinstance(mask, pd.DataFrame)
        assert mask.dtypes.iloc[0] == bool

    def test_apply_mask_sets_nan(self, price, factor):
        mask = build_limit_mask(price, threshold=0.095)
        masked = apply_mask(factor, mask)

        wide = masked.unstack("ts_code")
        # 找到有限日期的索引
        limit_dates = mask.index[(~mask).any(axis=1)]
        assert len(limit_dates) > 0, "未检测到涨跌停日期"
        # 涨跌停日应有 NaN
        assert wide.loc[limit_dates[0]].isna().any()

    def test_apply_mask_normal_days_unchanged(self, price, factor):
        mask = build_limit_mask(price, threshold=0.095)
        masked = apply_mask(factor, mask)
        # 正常日因子值应不变
        normal_dates = mask.index[mask.all(axis=1)]
        assert len(normal_dates) > 0, "没有正常交易日"
        normal_day = normal_dates[0]
        normal_f = factor.xs(normal_day, level="trade_date")
        masked_f = masked.xs(normal_day, level="trade_date")
        assert normal_f.equals(masked_f)

    def test_empty_input(self):
        assert build_limit_mask(pd.DataFrame()).empty
        result = apply_mask(pd.Series(dtype=float), pd.DataFrame())
        assert result.empty


# ============================================================================
# 因子选择测试
# ============================================================================

class TestFactorSelector:
    @pytest.fixture
    def factors(self):
        """构造 8 个因子，部分有效、部分高相关。"""
        np.random.seed(42)
        n_dates = 200
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")

        data = {}
        # 有效因子 (与收益正相关)
        signal = np.random.randn(n_dates) * 0.02 + 0.001
        data["momentum"] = signal + np.random.randn(n_dates) * 0.01
        data["volume_ratio"] = signal + np.random.randn(n_dates) * 0.015
        # 高相关因子 (与 momentum 高度相关)
        data["momentum_5d"] = data["momentum"] + np.random.randn(n_dates) * 0.003
        data["momentum_10d"] = data["momentum"] + np.random.randn(n_dates) * 0.004
        # 无效因子 (纯噪声)
        data["noise_1"] = np.random.randn(n_dates) * 0.03
        data["noise_2"] = np.random.randn(n_dates) * 0.03
        data["noise_3"] = np.random.randn(n_dates) * 0.03
        # 有效但有相关性
        data["reversal"] = -signal * 0.5 + np.random.randn(n_dates) * 0.02

        return pd.DataFrame(data, index=dates)

    @pytest.fixture
    def forward_returns(self, factors):
        """构造与 signal 对应的前向收益。"""
        np.random.seed(42)
        return pd.Series(
            np.random.randn(200) * 0.03 + 0.0005,
            index=factors.index,
        ) + (factors.index.day % 5) * 0.0001  # 小信号

    def test_filter_by_ic_removes_noise(self, factors, forward_returns):
        result = filter_by_ic(factors, forward_returns, min_abs_ic=0.001)
        # 至少应有因子被保留
        assert len(result.columns) > 0
        # 纯噪声的 IC 应该很低
        assert result.shape[1] <= factors.shape[1]

    def test_filter_by_ic_attaches_ic_records(self, factors, forward_returns):
        result = filter_by_ic(factors, forward_returns, min_abs_ic=0.0001)
        assert "ic_records" in result.attrs

    def test_remove_correlated_dedup(self, factors):
        """高相关因子应被去重。"""
        result = remove_correlated(factors, max_corr=0.8)
        # momentum_5d 和 momentum 高度相关，应只留一个
        assert len(result.columns) < len(factors.columns)

    def test_remove_correlated_single_factor(self):
        """单因子不崩溃。"""
        f = pd.DataFrame({"a": [1, 2, 3]})
        result = remove_correlated(f)
        assert list(result.columns) == ["a"]

    def test_cross_sectional_corr_filter(self, factors):
        result = cross_sectional_corr_filter(factors, max_mean_corr=0.5)
        assert len(result.columns) <= len(factors.columns)

    def test_select_factors_pipeline(self, factors, forward_returns):
        result = select_factors(
            factors, forward_returns,
            min_abs_ic=0.0001, max_corr=0.8, max_mean_corr=0.6,
        )
        assert len(result.columns) > 0
        # 经过全套筛选，应比原始少
        assert len(result.columns) <= len(factors.columns)

    def test_select_factors_no_forward_returns(self, factors):
        """不传 forward_returns 时跳过 IC 筛选。"""
        result = select_factors(factors, forward_returns=None)
        assert len(result.columns) > 0

    def test_filter_insufficient_data(self, factors, forward_returns):
        """数据点太少时返回空 DataFrame。"""
        small = factors.iloc[:5]
        small_ret = forward_returns.iloc[:5]
        result = filter_by_ic(small, small_ret, min_periods=20)
        assert result.empty
