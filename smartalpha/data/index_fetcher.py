"""指数数据获取器 — 沪深300、中证500、中证1000等市场基准指数。

为 BETA 计算、基准对比、市场择时提供数据基础。

数据源: Tushare (index_daily) 为主，AKShare 为 fallback。
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 常用市场指数
BENCHMARK_INDICES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "000016.SH": "上证50",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
    "000001.SH": "上证综指",
}


class IndexFetcher:
    """指数日线数据获取器。

    使用示例:
        fetcher = IndexFetcher()
        hsb = fetcher.fetch("000300.SH", "20200101", "20260730")
        print(hsb.head())
    """

    def __init__(self, token: Optional[str] = None) -> None:
        self._tushare = None
        self._akshare = None
        self._token = token

    def fetch(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
        prefer_tushare: bool = True,
    ) -> pd.DataFrame:
        """获取指数日线数据。

        Args:
            index_code: 指数代码 (如 "000300.SH")。
            start_date: 开始日期 YYYYMMDD。
            end_date: 结束日期 YYYYMMDD。
            prefer_tushare: True=优先 Tushare, False=优先 AKShare。

        Returns:
            统一格式的日线 DataFrame，含 close/change/pct_chg。
        """
        if prefer_tushare:
            df = self._fetch_from_tushare(index_code, start_date, end_date)
            if not df.empty:
                return df
            df = self._fetch_from_akshare(index_code, start_date, end_date)
            return df
        else:
            df = self._fetch_from_akshare(index_code, start_date, end_date)
            if not df.empty:
                return df
            df = self._fetch_from_tushare(index_code, start_date, end_date)
            return df

    def fetch_all_benchmarks(
        self, start_date: str, end_date: str
    ) -> dict[str, pd.DataFrame]:
        """批量获取所有基准指数数据。

        Returns:
            {index_code: DataFrame} 字典。
        """
        results = {}
        for i, (code, name) in enumerate(BENCHMARK_INDICES.items()):
            logger.info(f"获取 {name} ({code}) ...")
            df = self.fetch(code, start_date, end_date)
            if not df.empty:
                results[code] = df
            if i < len(BENCHMARK_INDICES) - 1:
                time.sleep(0.3)
        return results

    # ------------------------------------------------------------------
    # Tushare 实现
    # ------------------------------------------------------------------

    def _fetch_from_tushare(
        self, index_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """通过 Tushare 获取指数日线。"""
        try:
            from .fetcher import TushareFetcher
            if self._tushare is None:
                self._tushare = TushareFetcher(token=self._token)
        except Exception:
            logger.warning("TushareFetcher 初始化失败，指数数据不可用")
            return pd.DataFrame()

        try:
            raw = self._tushare._pro.index_daily(
                ts_code=index_code,
                start_date=start_date,
                end_date=end_date,
                fields="ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
            )
        except Exception as e:
            logger.warning(f"Tushare 指数 {index_code} 失败: {e}")
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        raw = raw.sort_values("trade_date").reset_index(drop=True)
        raw["trade_date"] = raw["trade_date"].astype(str)
        return raw

    # ------------------------------------------------------------------
    # AKShare 实现
    # ------------------------------------------------------------------

    def _fetch_from_akshare(
        self, index_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """通过 AKShare 获取指数日线。"""
        try:
            import akshare as ak
            if self._akshare is None:
                self._akshare = ak
        except ImportError:
            logger.warning("AKShare 未安装")
            return pd.DataFrame()

        # AKShare 指数代码映射
        ak_code_map = {
            "000300.SH": "sh000300",   # 沪深300
            "000905.SH": "sh000905",   # 中证500
            "000852.SH": "sh000852",   # 中证1000
            "000016.SH": "sh000016",   # 上证50
            "399006.SZ": "sz399006",   # 创业板指
            "000688.SH": "sh000688",   # 科创50
            "000001.SH": "sh000001",   # 上证综指
        }

        ak_code = ak_code_map.get(index_code, "")
        if not ak_code:
            # 尝试自动推测
            if ".SH" in index_code:
                ak_code = f"sh{index_code.replace('.SH', '')}"
            else:
                ak_code = f"sz{index_code.replace('.SZ', '')}"

        try:
            raw = ak.stock_zh_index_daily(symbol=ak_code)
        except Exception as e:
            logger.warning(f"AKShare 指数 {index_code} 失败: {e}")
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        # 统一列名
        result = pd.DataFrame()
        result["ts_code"] = index_code
        result["trade_date"] = raw["date"].astype(str).str.replace("-", "")
        result["open"] = pd.to_numeric(raw["open"], errors="coerce")
        result["high"] = pd.to_numeric(raw["high"], errors="coerce")
        result["low"] = pd.to_numeric(raw["low"], errors="coerce")
        result["close"] = pd.to_numeric(raw["close"], errors="coerce")
        result["vol"] = pd.to_numeric(raw["volume"], errors="coerce")
        result["amount"] = pd.to_numeric(raw.get("amount", 0), errors="coerce")

        # 过滤日期范围
        result = result[
            (result["trade_date"] >= start_date) & (result["trade_date"] <= end_date)
        ]
        return result.sort_values("trade_date").reset_index(drop=True)


def get_market_returns(
    index_code: str = "000300.SH",
    start_date: str = "20200101",
    end_date: str = "20260730",
) -> pd.Series:
    """便捷函数：获取市场指数日收益率序列。

    Args:
        index_code: 指数代码，默认沪深300。
        start_date: 开始日期。
        end_date: 结束日期。

    Returns:
        日收益率 Series (index=date)。
    """
    fetcher = IndexFetcher()
    df = fetcher.fetch(index_code, start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float)

    df = df.sort_values("trade_date")
    market_ret = pd.Series(
        df["close"].values,
        index=pd.to_datetime(df["trade_date"]),
    ).pct_change()
    return market_ret.rename(index_code)
