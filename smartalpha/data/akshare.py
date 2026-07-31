"""AKShare 数据获取器 — 完全免费、无需Token、全历史深度。

接口与 TushareFetcher 保持一致，上层代码无需修改。
AKShare 通过爬取东方财富等公开数据源获取数据，免费但较慢。

列名映射 (AKShare → 统一格式):
    日期 → trade_date, 开盘 → open, 最高 → high, 最低 → low
    收盘 → close, 成交量 → vol, 成交额 → amount
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AKShareFetcher:
    """A股日线数据获取器 (基于 AKShare)。

    使用示例:
        fetcher = AKShareFetcher()
        df = fetcher.daily("000001.SZ", "20240101", "20240725")

    注意: AKShare 免费但受东方财富反爬限制，单次请求约 3-5 秒。
          批量下载 5000 只股票约需 4-6 小时。
    """

    # AKShare 列名 → 统一格式 映射
    COLUMN_MAP = {
        "日期": "trade_date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "vol",
        "成交额": "amount",
    }

    def __init__(self) -> None:
        """初始化。无需Token。"""
        try:
            import akshare as ak
            self._ak = ak
        except ImportError:
            raise ImportError(
                "akshare 未安装。运行: pip install akshare"
            )

    # ------------------------------------------------------------------
    # A股日线行情
    # ------------------------------------------------------------------

    def daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取单只股票日线数据。

        Args:
            ts_code: Tushare格式股票代码 (如 "000001.SZ")。
            start_date: 开始日期 YYYYMMDD。
            end_date: 结束日期 YYYYMMDD。

        Returns:
            统一格式的日线DataFrame。
        """
        ak_symbol = self._to_akshare_code(ts_code)
        ak_start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
        ak_end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:8]}"

        try:
            raw = self._ak.stock_zh_a_hist(
                symbol=ak_symbol,
                period="daily",
                start_date=ak_start,
                end_date=ak_end,
                adjust="hfq",  # 后复权 (消除前复权 look-ahead bias)
            )
        except Exception as e:
            raise RuntimeError(f"AKShare获取 {ts_code} 失败: {e}")

        if raw is None or raw.empty:
            return pd.DataFrame()

        return self._normalize(raw, ts_code)

    def daily_batch(
        self,
        ts_codes: List[str],
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """批量获取多只股票日线数据。

        Args:
            ts_codes: 股票代码列表 (Tushare格式)。
            start_date: 开始日期 YYYYMMDD。
            end_date: 结束日期 YYYYMMDD。

        Returns:
            合并后的DataFrame。
        """
        frames = []
        for i, code in enumerate(ts_codes):
            try:
                df = self.daily(code, start_date, end_date)
                if not df.empty:
                    frames.append(df)
            except Exception as e:
                logger.warning(f"AKShare 获取 {code} 失败: {e}")

            # 反爬延迟: 每次请求间隔 3-5 秒
            if i < len(ts_codes) - 1:
                time.sleep(3.0)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["trade_date"] = result["trade_date"].astype(str)
        return result.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 股票基础信息
    # ------------------------------------------------------------------

    def stock_list(self) -> pd.DataFrame:
        """获取A股股票基础信息（含退市股票，消除幸存者偏差）。

        Returns:
            统一格式的DataFrame (ts_code, name, industry, list_date, list_status)。
            网络错误时返回空DataFrame。

        注意: AKShare 的股票快照接口仅返回当前上市股票。
              退市股票需通过 stock_zh_a_hist() 按已知退市代码获取历史数据。
              此处返回的 list_status 列标注股票状态 (L=上市/D=退市/P=暂停)。
        """
        try:
            raw = self._ak.stock_zh_a_spot_em()
        except Exception as e:
            import warnings
            warnings.warn(f"AKShare stock_list 失败 (网络问题?): {e}")
            return pd.DataFrame()

        if raw is None or raw.empty:
            return pd.DataFrame()

        result = pd.DataFrame()
        result["ts_code"] = raw["代码"].apply(self._to_tushare_code)
        result["name"] = raw["名称"]
        result["industry"] = raw.get("所属行业", "")
        result["list_date"] = ""
        result["list_status"] = "L"  # 当前快照仅含上市状态

        # 尝试补充退市股票列表 (stock_info_a_code_name 可能包含部分退市)
        try:
            all_info = self._ak.stock_info_a_code_name()
            if all_info is not None and not all_info.empty:
                all_codes = set(all_info["code"].apply(self._to_tushare_code))
                existing = set(result["ts_code"])
                extra_codes = all_codes - existing
                if extra_codes:
                    extra = pd.DataFrame({
                        "ts_code": sorted(extra_codes),
                        "name": "",
                        "industry": "",
                        "list_date": "",
                        "list_status": "D",  # 不在快照中的标记为退市/暂停
                    })
                    result = pd.concat([result, extra], ignore_index=True)
        except Exception:
            logger.warning("退市列表补充失败，将仅使用在交易股票列表")

        return result

    # ------------------------------------------------------------------
    # 代码格式转换
    # ------------------------------------------------------------------

    @staticmethod
    def _to_akshare_code(ts_code: str) -> str:
        """Tushare格式 → AKShare格式。

        "000001.SZ" → "000001"
        "600000.SH" → "600000"
        """
        if "." in ts_code:
            return ts_code.split(".")[0]
        return ts_code

    @staticmethod
    def _to_tushare_code(ak_code: str) -> str:
        """AKShare格式 → Tushare格式。

        6位数字 → 加后缀 .SH/.SZ
        """
        code = str(ak_code).zfill(6)
        if code.startswith(("6", "9")):
            return f"{code}.SH"
        elif code.startswith(("0", "3", "2")):
            return f"{code}.SZ"
        else:
            return f"{code}.SH"

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _normalize(self, raw: pd.DataFrame, ts_code: str) -> pd.DataFrame:
        """将 AKShare 原始数据转换为统一格式。

        Args:
            raw: AKShare 返回的原始 DataFrame。
            ts_code: 股票代码。

        Returns:
            统一格式的 DataFrame。
        """
        df = pd.DataFrame()
        df["ts_code"] = ts_code

        for ak_col, our_col in self.COLUMN_MAP.items():
            if ak_col in raw.columns:
                df[our_col] = raw[ak_col]
            else:
                df[our_col] = None

        # 日期格式: "2024-01-15" → "20240115"
        if "日期" in raw.columns:
            df["trade_date"] = raw["日期"].astype(str).str.replace("-", "")
        elif "trade_date" in df.columns:
            pass  # 已有

        # 确保数值类型
        for col in ["open", "high", "low", "close", "vol", "amount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.dropna(subset=["close"]).reset_index(drop=True)
