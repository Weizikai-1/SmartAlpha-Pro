"""行业分类获取器 — 构建 股票→行业 映射表。

为因子中性化、行业集中度风控提供行业分类基础。

数据源优先级:
1. Tushare stock_basic 中的 industry 字段 (最快)
2. AKShare stock_zh_a_spot_em 中的行业信息 (免费)
3. 内置 fallback: 基于股票代码前缀的粗分类

所有方法最终返回 dict[str, str]: stock_code → industry_name
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from .akshare import AKShareFetcher

logger = logging.getLogger(__name__)


class IndustryFetcher:
    """行业分类获取器。

    使用示例:
        fetcher = IndustryFetcher()
        industry_map = fetcher.build_map()
        print(industry_map["000001.SZ"])  # → "银行"
    """

    def __init__(self, token: Optional[str] = None) -> None:
        self._token = token
        self._tushare = None

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def build_map(
        self,
        prefer_tushare: bool = True,
    ) -> dict[str, str]:
        """构建完整 股票→行业 映射表。

        Args:
            prefer_tushare: True=优先 Tushare, False=优先 AKShare。

        Returns:
            {stock_code: industry_name} 字典。
        """
        if prefer_tushare:
            result = self._from_tushare()
            if result:
                return result
            result = self._from_akshare()
            if result:
                return result
        else:
            result = self._from_akshare()
            if result:
                return result
            result = self._from_tushare()
            if result:
                return result

        # 最终 fallback
        logger.warning("无法获取行业分类，使用内置粗分类")
        return self._builtin_fallback()

    def get_industry_map(
        self,
        ts_codes: list[str],
        prefer_tushare: bool = True,
    ) -> dict[str, str]:
        """获取指定股票列表的行业映射。

        Args:
            ts_codes: 股票代码列表。
            prefer_tushare: 数据源偏好。

        Returns:
            {stock_code: industry_name} 字典。缺失时用代码前缀粗分类兜底。
        """
        full_map = self.build_map(prefer_tushare)
        return {k: full_map.get(k) or self.coarse_classify(k) for k in ts_codes}

    # ------------------------------------------------------------------
    # Tushare 数据源
    # ------------------------------------------------------------------

    def _from_tushare(self) -> dict[str, str]:
        """从 Tushare stock_basic 获取行业分类。"""
        try:
            from .fetcher import TushareFetcher
            if self._tushare is None:
                self._tushare = TushareFetcher(token=self._token)
        except Exception as e:
            logger.warning(f"Tushare 不可用: {e}")
            return {}

        try:
            df = self._tushare._pro.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,industry",
            )
        except Exception as e:
            logger.warning(f"Tushare stock_basic 失败: {e}")
            return {}

        if df is None or df.empty:
            return {}

        result = {}
        for _, row in df.iterrows():
            code = row.get("ts_code", "")
            ind = row.get("industry", "")
            if code and ind and pd.notna(ind):
                result[code] = str(ind)

        logger.info(f"Tushare 行业: {len(result)} 只股票")
        return result

    # ------------------------------------------------------------------
    # AKShare 数据源
    # ------------------------------------------------------------------

    def _from_akshare(self) -> dict[str, str]:
        """从 AKShare 获取行业分类。"""
        try:
            import akshare as ak
        except ImportError:
            return {}

        try:
            raw = ak.stock_zh_a_spot_em()
        except Exception as e:
            logger.warning(f"AKShare stock_zh_a_spot_em 失败: {e}")
            return {}

        if raw is None or raw.empty:
            return {}

        result = {}
        for _, row in raw.iterrows():
            code = AKShareFetcher._to_tushare_code(row.get("代码", ""))
            industry = row.get("所属行业", "")
            if code and industry and pd.notna(industry):
                result[code] = str(industry)

        logger.info(f"AKShare 行业: {len(result)} 只股票")
        return result

    # ------------------------------------------------------------------
    # 内置粗分类 (fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _builtin_fallback() -> dict[str, str]:
        """基于股票代码前缀的内置粗分类。

        规则:
        - 00/30 → 深市
        - 60 → 沪市主板
        - 68 → 科创板
        - 其他 → 未知

        这仅是极端情况下的应急分类，不应作为生产级方案。
        """
        mapping = {}
        # 这里返回空映射 — 调用方负责处理 "未知"
        return mapping

    @staticmethod
    def coarse_classify(code: str) -> str:
        """对单个股票代码做粗分类（应急用）。"""
        if code.startswith(("600", "601", "603", "605")):
            return "沪市主板"
        elif code.startswith("000") or code.startswith("001"):
            return "深市主板"
        elif code.startswith("002"):
            return "中小板"
        elif code.startswith("300") or code.startswith("301"):
            return "创业板"
        elif code.startswith("688"):
            return "科创板"
        elif code.startswith("430") or code.startswith("8"):
            return "北交所"
        else:
            return "未知"


def load_industry_map_from_cache(cache_path: str = "data/industry_map.parquet") -> dict[str, str]:
    """从本地缓存加载行业映射。

    Args:
        cache_path: Parquet 文件路径。

    Returns:
        行业映射字典，文件不存在则返回空。
    """
    import os
    if not os.path.exists(cache_path):
        return {}
    try:
        df = pd.read_parquet(cache_path)
        return dict(zip(df["ts_code"], df["industry"]))
    except Exception:
        logger.warning(f"行业缓存读取失败: {cache_path}")
        return {}


def save_industry_map_to_cache(
    industry_map: dict[str, str],
    cache_path: str = "data/industry_map.parquet",
) -> None:
    """保存行业映射到本地缓存。

    Args:
        industry_map: {stock_code: industry_name} 字典。
        cache_path: 输出路径。
    """
    import os
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df = pd.DataFrame({
        "ts_code": list(industry_map.keys()),
        "industry": list(industry_map.values()),
    })
    df.to_parquet(cache_path, index=False)
