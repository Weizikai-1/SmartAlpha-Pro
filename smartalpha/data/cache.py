"""本地Parquet缓存 — 避免重复调用API。

自动将tushare返回的DataFrame缓存到本地Parquet文件。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import pandas as pd


class DataCache:
    """本地Parquet文件缓存。

    使用示例:
        cache = DataCache("~/.smartalpha/cache")
        df = cache.get("daily_000001.SZ_20240101_20240131")
        if df is None:
            df = fetch_from_api(...)
            cache.put("daily_000001.SZ_20240101_20240131", df)
    """

    def __init__(self, cache_dir: str = "~/.smartalpha/cache") -> None:
        """初始化缓存。

        Args:
            cache_dir: 缓存目录路径。
        """
        self._dir = Path(cache_dir).expanduser().resolve()
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """读取缓存。

        Args:
            key: 缓存键。

        Returns:
            缓存的DataFrame，不存在则返回None。
        """
        path = self._path(key)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"缓存读取失败: {path}, {e}")
                path.unlink(missing_ok=True)
        return None

    def put(self, key: str, df: pd.DataFrame) -> None:
        """写入缓存。

        Args:
            key: 缓存键。
            df: 要缓存的数据。
        """
        df.to_parquet(self._path(key), index=False)

    def clear(self) -> None:
        """清空全部缓存。"""
        for f in self._dir.glob("*.parquet"):
            f.unlink()

    def stats(self) -> dict:
        """缓存统计信息。"""
        files = list(self._dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)
        return {"file_count": len(files), "total_size_mb": total_size / (1024 * 1024)}

    def _path(self, key: str) -> Path:
        """将key转换为文件路径。"""
        safe = hashlib.md5(key.encode()).hexdigest()
        return self._dir / f"{safe}.parquet"
