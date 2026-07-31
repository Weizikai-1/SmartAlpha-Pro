"""统一数据加载器 — 多数据源自动切换 + 缓存 + 质量检查。

数据源优先级: Tushare → AKShare (免费fallback)
所有数据存入同一缓存，上层代码无需关心数据来源。

使用示例:
    loader = DataLoader()
    df = loader.load_daily(["000001.SZ", "000002.SZ"], "20240101", "20240131")
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from .cache import DataCache
from .quality import DataQualityChecker, QualityReport

logger = logging.getLogger(__name__)


class DataLoader:
    """统一数据加载器，多数据源自动切换。

    数据源:
    - Tushare Pro (需Token，快速，有限额)
    - AKShare (免费，慢，无限制)

    自动选择策略:
    1. 优先从缓存读取
    2. 缓存未命中时尝试 Tushare
    3. Tushare不可用时自动切换到 AKShare
    """

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: str = "~/.smartalpha/cache",
        prefer_akshare: bool = False,
    ) -> None:
        """初始化。

        Args:
            token: tushare token，默认从环境变量读取。
            cache_dir: 缓存目录。
            prefer_akshare: True=优先使用AKShare(不计额度), False=优先Tushare。
        """
        self.cache = DataCache(cache_dir=cache_dir)
        self.quality = DataQualityChecker()
        self._prefer_akshare = prefer_akshare

        # 延迟初始化数据源
        self._tushare = None
        self._akshare = None
        self._tushare_available = None
        self._akshare_available = None

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def load_daily(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        check_quality: bool = True,
    ) -> pd.DataFrame:
        """加载日线数据（优先缓存，其次API）。

        Args:
            ts_codes: 股票代码列表。
            start_date: 开始日期 YYYYMMDD。
            end_date: 结束日期 YYYYMMDD。
            use_cache: 是否使用本地缓存。
            check_quality: 是否执行质量检查。

        Returns:
            日线行情DataFrame。

        Raises:
            RuntimeError: 所有数据源均不可用时。
        """
        # 先查缓存
        cache_key = f"daily_{'_'.join(ts_codes)}_{start_date}_{end_date}"
        if use_cache:
            df = self.cache.get(cache_key)
            if df is not None:
                return df

        # 从API获取
        df = self._fetch_from_best_source(ts_codes, start_date, end_date)
        if df.empty:
            return df

        # 写入缓存
        if use_cache:
            self.cache.put(cache_key, df)

        # 质量检查
        if check_quality:
            report = self.quality.check(df)
            if not report.passed:
                raise ValueError(f"数据质量检查不通过:\n{report}")

        return df

    def load_stock_list(self, use_cache: bool = True) -> pd.DataFrame:
        """加载A股股票列表。

        Args:
            use_cache: 是否使用缓存。

        Returns:
            股票基础信息DataFrame。
        """
        cache_key = "stock_list"
        if use_cache:
            df = self.cache.get(cache_key)
            if df is not None:
                return df

        fetcher = self._get_available_fetcher()
        df = fetcher.stock_list()
        if use_cache and not df.empty:
            self.cache.put(cache_key, df)
        return df

    # ------------------------------------------------------------------
    # 单只股票缓存 (用于批量下载)
    # ------------------------------------------------------------------

    def cache_single_stock(
        self, ts_code: str, start_date: str, end_date: str
    ) -> int:
        """获取单只股票数据并缓存（返回行数）。

        用于批量下载场景，每只股票独立缓存。

        Args:
            ts_code: 股票代码。
            start_date: 开始日期。
            end_date: 结束日期。

        Returns:
            缓存的行数。0表示失败。
        """
        cache_key = f"daily_{[ts_code]}_{start_date}_{end_date}"

        # 已有缓存则跳过
        if self.cache.get(cache_key) is not None:
            return -1  # 跳过

        fetcher = self._get_available_fetcher()
        df = fetcher.daily(ts_code, start_date, end_date)
        if not df.empty:
            self.cache.put(cache_key, df)
        return len(df)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _fetch_from_best_source(
        self, ts_codes: List[str], start_date: str, end_date: str
    ) -> pd.DataFrame:
        """从最佳可用数据源获取数据。"""
        fetcher = self._get_available_fetcher()
        return fetcher.daily_batch(ts_codes, start_date, end_date)

    def _get_available_fetcher(self):
        """获取可用的数据获取器。

        Tushare优先（更快），AKShare作为fallback。
        """
        # 按优先级尝试
        if not self._prefer_akshare:
            if self._tushare_available is None:
                self._try_init_tushare()
            if self._tushare_available:
                return self._tushare

        if self._akshare_available is None:
            self._try_init_akshare()
        if self._akshare_available:
            return self._akshare

        # 如果偏好AKShare但Tushare优先时Tushare不可用
        if self._prefer_akshare and self._tushare_available is None:
            self._try_init_tushare()
        if self._tushare_available:
            return self._tushare

        raise RuntimeError(
            "所有数据源均不可用。请配置 Tushare Token 或安装 akshare (pip install akshare)"
        )

    def _try_init_tushare(self) -> None:
        """尝试初始化 Tushare。"""
        try:
            from .fetcher import TushareFetcher as TF
            self._tushare = TF()
            self._tushare_available = True
        except Exception as e:
            logger.warning(f"Tushare 不可用: {e}")
            self._tushare_available = False

    def _try_init_akshare(self) -> None:
        """尝试初始化 AKShare。"""
        try:
            from .akshare import AKShareFetcher as AF
            self._akshare = AF()
            self._akshare_available = True
        except Exception as e:
            logger.warning(f"AKShare 不可用: {e}")
            self._akshare_available = False

    @property
    def active_source(self) -> str:
        """当前活跃的数据源名称。"""
        if self._tushare_available:
            return "Tushare"
        if self._akshare_available:
            return "AKShare"
        return "无"
