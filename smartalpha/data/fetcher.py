"""Tushare API 封装 — A股日线数据获取器。

依赖: tushare>=1.4.0, python-dotenv
权限: 需要tushare pro token (环境变量 TUSHARE_TOKEN)
限制: 新用户100积分，daily接口每次最多6000条
"""

from __future__ import annotations

import os
import random
import time
from typing import List, Optional

import pandas as pd
import tushare as ts
from dotenv import load_dotenv


class TushareError(Exception):
    """Tushare API 错误。"""


class TushareFetcher:
    """A股日线数据获取器。

    使用示例:
        fetcher = TushareFetcher()
        df = fetcher.daily("000001.SZ", "20240101", "20240131")
    """

    # 单次调用最大返回行数
    MAX_ROWS = 6000

    def __init__(self, token: Optional[str] = None) -> None:
        """初始化。

        Args:
            token: tushare token，为None时从环境变量 TUSHARE_TOKEN 读取。
        """
        if token is None:
            load_dotenv()
            token = os.getenv("TUSHARE_TOKEN")
        if not token or token == "your_token_here":
            raise TushareError(
                "未配置 TUSHARE_TOKEN。请在 .env 文件中设置，"
                "或访问 https://tushare.pro 注册获取。"
            )
        # 通过环境变量传递 token (避免 ts.set_token() 写 tk.csv 的权限问题)
        os.environ["TUSHARE_TOKEN"] = token
        self._pro = ts.pro_api(token)

    # ------------------------------------------------------------------
    # A股日线行情
    # ------------------------------------------------------------------

    def daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取单只股票日线数据（含复权因子）。

        Args:
            ts_code: 股票代码 (如 "000001.SZ")。
            start_date: 开始日期 YYYYMMDD。
            end_date: 结束日期 YYYYMMDD。
            fields: 返回字段，默认请求 adj_factor 用于后复权计算。

        Returns:
            按日期升序排列的日线数据，含 adj_factor 列。
            后复权收盘价 = close × adj_factor。
        """
        # 默认请求复权因子（消除前复权 look-ahead bias）
        if fields is None:
            fields = "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount,adj_factor"
        df = self._fetch_with_retry("daily", {
            "ts_code": ts_code,
            "start_date": start_date,
            "end_date": end_date,
            "fields": fields,
        })
        if not df.empty and "trade_date" in df.columns:
            df = df.sort_values("trade_date").reset_index(drop=True)
        return df

    def daily_batch(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[str] = None,
    ) -> pd.DataFrame:
        """批量获取多只股票日线数据。

        Args:
            ts_codes: 股票代码列表。
            start_date: 开始日期 YYYYMMDD。
            end_date: 结束日期 YYYYMMDD。
            fields: 返回字段，逗号分隔。

        Returns:
            合并后的DataFrame，按日期、股票代码排序。
        """
        frames = []
        for i, code in enumerate(ts_codes):
            df = self.daily(code, start_date, end_date, fields)
            df["ts_code"] = code  # 确保有股票代码列
            frames.append(df)
            # 避免触发速率限制
            if i < len(ts_codes) - 1:
                time.sleep(0.3)
        result = pd.concat(frames, ignore_index=True)
        result["trade_date"] = result["trade_date"].astype(str)
        return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 股票基础信息
    # ------------------------------------------------------------------

    def stock_list(self, exchange: str = "") -> pd.DataFrame:
        """获取A股股票基础信息（含退市+暂停，消除幸存者偏差）。

        Args:
            exchange: 交易所代码 (SSE=上交所, SZSE=深交所, ""=全部)。

        Returns:
            股票基础信息DataFrame，包含上市/退市/暂停全部状态。
        """
        # 不传 list_status → 获取 L(上市)/D(退市)/P(暂停) 全状态
        return self._fetch_with_retry("stock_basic", {
            "exchange": exchange,
            "fields": "ts_code,name,area,industry,list_date,list_status",
        })

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def trade_cal(
        self, start_date: str, end_date: str, exchange: str = "SSE"
    ) -> pd.DataFrame:
        """获取交易日历。

        Args:
            start_date: 开始日期 YYYYMMDD。
            end_date: 结束日期 YYYYMMDD。
            exchange: 交易所 (SSE/SZSE)。

        Returns:
            包含 is_open 列的DataFrame。
        """
        return self._fetch_with_retry("trade_cal", {
            "exchange": exchange,
            "start_date": start_date,
            "end_date": end_date,
            "is_open": "1",
        })

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self, api_name: str, params: dict, max_retries: int = 3
    ) -> pd.DataFrame:
        """带指数退避 + jitter 的 API 调用重试。

        重试策略:
            wait = min(base * 2^attempt, 30) + random(0, jitter)
            其中 base=1s, jitter=0.5s, 最大等待 30s。
        """
        params = {k: v for k, v in params.items() if v is not None}
        base_wait = 1.0
        max_wait = 30.0
        jitter = 0.5

        for attempt in range(max_retries):
            try:
                result = self._pro.query(api_name, **params)
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
                if result is None:
                    return pd.DataFrame()
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    raise TushareError(
                        f"API {api_name} 调用失败 ({max_retries}次重试): {e}"
                    )
                # 指数退避: base * 2^attempt, 上限 max_wait
                wait = min(base_wait * (2 ** attempt), max_wait)
                # jitter: ±0.5s 随机化，避免惊群效应
                wait += random.uniform(-jitter, jitter)
                wait = max(0.1, wait)  # 最低 0.1s
                time.sleep(wait)
        return pd.DataFrame()
