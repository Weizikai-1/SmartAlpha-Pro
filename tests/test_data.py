"""数据层测试 — DataCache, DataQualityChecker, DataLoader。

测试: 缓存读写、质量检查、加载器流程（mock fetcher）。
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from smartalpha.data.cache import DataCache
from smartalpha.data.quality import DataQualityChecker, QualityReport
from smartalpha.data.loader import DataLoader


# ============================================================================
# DataCache 测试
# ============================================================================

class TestDataCache:
    @pytest.fixture
    def cache_dir(self):
        d = tempfile.mkdtemp()
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def dc(self, cache_dir):
        return DataCache(cache_dir=cache_dir)

    def test_put_and_get(self, dc):
        df = pd.DataFrame({"close": [10.0, 11.0, 12.0]})
        dc.put("test_key", df)
        result = dc.get("test_key")
        assert result is not None
        assert len(result) == 3
        assert result["close"].tolist() == [10.0, 11.0, 12.0]

    def test_get_miss(self, dc):
        assert dc.get("nonexistent") is None

    def test_get_corrupted_file(self, dc):
        """损坏的parquet文件应返回None并被删除。"""
        path = dc._path("corrupt")
        path.write_text("not a parquet file")
        result = dc.get("corrupt")
        assert result is None
        assert not path.exists()

    def test_clear(self, dc):
        dc.put("a", pd.DataFrame({"x": [1]}))
        dc.put("b", pd.DataFrame({"x": [2]}))
        dc.clear()
        stats = dc.stats()
        assert stats["file_count"] == 0

    def test_stats(self, dc):
        dc.put("a", pd.DataFrame({"x": [1]}))
        dc.put("b", pd.DataFrame({"x": [2, 3]}))
        stats = dc.stats()
        assert stats["file_count"] == 2
        assert stats["total_size_mb"] >= 0

    def test_cache_dir_created(self, cache_dir):
        sub = os.path.join(cache_dir, "sub", "deep")
        DataCache(cache_dir=sub)
        assert Path(sub).exists()

    def test_overwrite(self, dc):
        df1 = pd.DataFrame({"x": [1]})
        df2 = pd.DataFrame({"x": [100]})
        dc.put("key", df1)
        dc.put("key", df2)
        result = dc.get("key")
        assert result["x"].iloc[0] == 100


# ============================================================================
# DataQualityChecker 测试
# ============================================================================

class TestDataQuality:
    @pytest.fixture
    def checker(self):
        return DataQualityChecker()

    @pytest.fixture
    def valid_df(self):
        return pd.DataFrame({
            "trade_date": ["20240101", "20240102", "20240103"],
            "open": [10.0, 11.0, 12.0],
            "high": [12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0],
            "close": [11.0, 12.0, 13.0],
            "vol": [1000, 2000, 3000],
        })

    def test_valid_data_passes(self, checker, valid_df):
        report = checker.check(valid_df)
        assert report.passed
        assert all(report.checks.values())

    def test_missing_columns(self, checker):
        df = pd.DataFrame({"open": [10, 11]})
        report = checker.check(df)
        assert not report.passed
        assert not report.checks["columns"]

    def test_high_missing_ratio(self, checker):
        """缺失值超过10%阈值。"""
        n = 100
        data = {"open": [10.0] * n, "high": [12.0] * n, "low": [9.0] * n, "close": [np.nan] * 20 + [11.0] * 80}
        df = pd.DataFrame(data)
        report = checker.check(df)
        assert not report.passed
        assert not report.checks["missing"]

    def test_price_out_of_range(self, checker):
        """价格超出合理范围。"""
        df = pd.DataFrame({
            "open": [0.005, 10.0],
            "high": [0.005, 10.0],
            "low": [0.005, 10.0],
            "close": [0.005, 10.0],
            "vol": [100, 200],
        })
        report = checker.check(df)
        assert not report.passed
        assert not report.checks["price_range"]

    def test_date_not_sorted(self, checker):
        df = pd.DataFrame({
            "trade_date": ["20240103", "20240101", "20240102"],
            "open": [10.0, 11.0, 12.0],
            "high": [12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0],
            "close": [11.0, 12.0, 13.0],
            "vol": [1000, 2000, 3000],
        })
        report = checker.check(df)
        assert not report.passed
        assert not report.checks["date_sorted"]

    def test_empty_dataframe(self, checker):
        report = checker.check(pd.DataFrame())
        assert not report.passed
        assert not report.checks["missing"]

    def test_quality_report_warnings(self, checker):
        df = pd.DataFrame({"close": [10]})
        report = checker.check(df)
        assert len(report.warnings) > 0

    def test_quality_report_repr(self, checker, valid_df):
        report = checker.check(valid_df)
        r = repr(report)
        assert "PASS" in r or "FAIL" in r


# ============================================================================
# DataLoader 测试（mock fetcher）
# ============================================================================

# 模拟 TushareFetcher 和 AKShareFetcher 的返回数据
_MOCK_DAILY = pd.DataFrame({
    "trade_date": ["20240102", "20240103", "20240104"],
    "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
    "open": [10.0, 11.0, 12.0],
    "high": [12.0, 13.0, 14.0],
    "low": [9.0, 10.0, 11.0],
    "close": [11.0, 12.0, 13.0],
    "vol": [1e6, 2e6, 3e6],
    "amount": [1e7, 2e7, 3e7],
})

_MOCK_STOCK_LIST = pd.DataFrame({
    "ts_code": ["000001.SZ", "000002.SZ"],
    "name": ["平安银行", "万科A"],
    "industry": ["银行", "房地产"],
    "list_date": ["19910403", "19910129"],
})


class MockFetcher:
    """模拟数据获取器，返回固定数据。"""
    def daily(self, ts_code, start_date, end_date):
        return _MOCK_DAILY.copy()

    def daily_batch(self, ts_codes, start_date, end_date):
        dfs = []
        for code in ts_codes:
            d = _MOCK_DAILY.copy()
            d["ts_code"] = code
            dfs.append(d)
        return pd.concat(dfs, ignore_index=True)

    def stock_list(self):
        return _MOCK_STOCK_LIST.copy()


class TestDataLoader:
    @pytest.fixture
    def cache_dir(self):
        d = tempfile.mkdtemp()
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def loader(self, cache_dir):
        """创建 loader 并用 mock 替换内部 fetcher。"""
        l = DataLoader(cache_dir=cache_dir)
        mock = MockFetcher()
        l._tushare = mock
        l._tushare_available = True
        l._akshare_available = False
        return l

    def test_load_daily_basic(self, loader):
        df = loader.load_daily(
            ts_codes=["000001.SZ"],
            start_date="20240101",
            end_date="20240131",
            use_cache=False,
            check_quality=False,
        )
        assert len(df) > 0
        assert "close" in df.columns

    def test_load_daily_from_cache(self, loader):
        """第一次写入缓存，第二次从缓存读取。"""
        codes = ["000001.SZ"]
        # 先用 use_cache=True 加载并写入缓存
        df1 = loader.load_daily(codes, "20240101", "20240131", use_cache=True, check_quality=False)
        # loader 内部缓存键: f"daily_{'_'.join(ts_codes)}_{start_date}_{end_date}"
        cache_key = "daily_000001.SZ_20240101_20240131"
        saved = loader.cache.get(cache_key)
        assert saved is not None

    def test_load_stock_list(self, loader):
        df = loader.load_stock_list(use_cache=False)
        assert len(df) == 2
        assert "ts_code" in df.columns
        assert "平安银行" in df["name"].values

    def test_cache_single_stock_new(self, loader):
        rows = loader.cache_single_stock("000001.SZ", "20240101", "20240131")
        assert rows > 0  # 返回实际行数

    def test_cache_single_stock_skip_cached(self, loader):
        """缓存命中时返回 -1。"""
        loader.cache_single_stock("000001.SZ", "20240101", "20240131")
        rows = loader.cache_single_stock("000001.SZ", "20240101", "20240131")
        assert rows == -1  # 跳过

    def test_load_daily_quality_check_fail(self, loader):
        """质量检查不通过时抛异常。"""
        # load_daily 内部调 daily_batch，需要 mock 它
        loader._tushare.daily_batch = lambda codes, s, e: pd.DataFrame({"bad_col": [1]})
        with pytest.raises(ValueError, match="质量检查"):
            loader.load_daily(["000001.SZ"], "20240101", "20240131", use_cache=False)

    def test_active_source_tushare(self, loader):
        assert loader.active_source == "Tushare"

    def test_active_source_none(self, cache_dir):
        """所有数据源不可用时。"""
        l = DataLoader(cache_dir=cache_dir)
        l._tushare_available = False
        l._akshare_available = False
        assert l.active_source == "无"

    def test_active_source_akshare(self, cache_dir):
        l = DataLoader(cache_dir=cache_dir)
        l._tushare_available = False
        l._akshare_available = True
        assert l.active_source == "AKShare"

    def test_prefer_akshare(self, cache_dir):
        """prefer_akshare=True 时优先 AKShare。"""
        l = DataLoader(cache_dir=cache_dir, prefer_akshare=True)
        mock = MockFetcher()
        l._akshare = mock
        l._akshare_available = True
        l._tushare_available = False
        # 不会抛 RuntimeError，因为 AKShare 可用
        assert l.active_source == "AKShare"

    def test_all_sources_unavailable(self, cache_dir):
        """所有数据源不可用时抛 RuntimeError。"""
        l = DataLoader(cache_dir=cache_dir)
        l._tushare_available = False
        l._akshare_available = False
        with pytest.raises(RuntimeError, match="数据源"):
            l.load_daily(["000001.SZ"], "20240101", "20240131", use_cache=False)

    def test_fetch_from_best_source_tushare_first(self, loader):
        """Tushare 优先且可用时应返回 Tushare 数据。"""
        df = loader._fetch_from_best_source(["000001.SZ"], "20240101", "20240131")
        assert len(df) > 0

    def test_cache_single_stock_empty_result(self, loader):
        """fetcher 返回空 DataFrame 时返回 0。"""
        loader._tushare.daily = lambda *a, **kw: pd.DataFrame()
        rows = loader.cache_single_stock("000001.SZ", "20240101", "20240131")
        assert rows == 0

    def test_load_stock_list_empty_result(self, loader):
        """fetcher 返回空时 loader 不崩溃也不缓存。"""
        loader._tushare.stock_list = lambda: pd.DataFrame()
        df = loader.load_stock_list(use_cache=False)
        assert df.empty

    def test_load_daily_batch_multiple_stocks(self, loader):
        df = loader.load_daily(
            ts_codes=["000001.SZ", "000002.SZ"],
            start_date="20240101",
            end_date="20240131",
            use_cache=False,
            check_quality=False,
        )
        assert df["ts_code"].nunique() == 2
