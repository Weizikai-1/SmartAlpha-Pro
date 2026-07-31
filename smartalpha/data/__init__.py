"""smartalpha.data — 真实数据获取与缓存模块。"""

from .fetcher import TushareFetcher
from .akshare import AKShareFetcher
from .cache import DataCache
from .quality import DataQualityChecker
from .loader import DataLoader
from .index_fetcher import IndexFetcher, get_market_returns, BENCHMARK_INDICES
from .industry_fetcher import (
    IndustryFetcher,
    load_industry_map_from_cache,
    save_industry_map_to_cache,
)
from .panel_builder import PanelBuilder, build_panel_from_cache

__all__ = [
    "TushareFetcher",
    "AKShareFetcher",
    "DataCache",
    "DataQualityChecker",
    "DataLoader",
    "IndexFetcher",
    "get_market_returns",
    "BENCHMARK_INDICES",
    "IndustryFetcher",
    "load_industry_map_from_cache",
    "save_industry_map_to_cache",
    "PanelBuilder",
    "build_panel_from_cache",
]
