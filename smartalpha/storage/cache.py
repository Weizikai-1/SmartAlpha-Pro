"""LRU 缓存模块。

提供带统计信息的 LRU（Least Recently Used）缓存实现，
用于因子计算结果的缓存与复用。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Hashable, Optional


@dataclass
class CacheStats:
    """缓存统计信息。

    Attributes:
        hits: 缓存命中次数。
        misses: 缓存未命中次数。
        evictions: 因容量限制被驱逐的条目数。
        expirations: 因过期被移除的条目数。
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def total(self) -> int:
        """总访问次数。"""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """命中率。"""
        if self.total == 0:
            return 0.0
        return self.hits / self.total

    def reset(self) -> None:
        """重置所有统计数据。"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0


class LRUCache:
    """LRU 缓存实现。

    基于 OrderedDict 实现，支持容量限制与可选的过期时间（TTL）。
    提供缓存命中率统计，便于性能监控。

    使用示例::

        cache = LRUCache(max_size=128, ttl=60.0)
        cache.put("key", value)
        value = cache.get("key")
        print(cache.stats.hit_rate)

    Attributes:
        max_size: 最大缓存条目数。
        ttl: 条目生存时间（秒），0 表示永不过期。
        stats: 缓存统计信息。
    """

    def __init__(self, max_size: int = 256, ttl: float = 0.0) -> None:
        """初始化 LRU 缓存。

        Args:
            max_size: 最大缓存条目数。
            ttl: 过期时间（秒），0 表示永不过期。
        """
        self._max_size: int = max(1, max_size)
        self._ttl: float = ttl
        self._data: OrderedDict[Hashable, tuple[Any, float]] = OrderedDict()
        self._stats: CacheStats = CacheStats()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @property
    def max_size(self) -> int:
        return self._max_size

    @property
    def ttl(self) -> float:
        return self._ttl

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def get(self, key: Hashable) -> Optional[Any]:
        """获取缓存值。

        Args:
            key: 缓存键。

        Returns:
            缓存的值，若键不存在或已过期返回 None。
        """
        if key not in self._data:
            self._stats.misses += 1
            return None

        value, expire_at = self._data[key]
        now = time.time()

        if self._ttl > 0 and now > expire_at:
            del self._data[key]
            self._stats.expirations += 1
            self._stats.misses += 1
            return None

        # 移至末尾（标记为最近使用）
        self._data.move_to_end(key)
        self._stats.hits += 1
        return value

    def put(self, key: Hashable, value: Any) -> None:
        """存入缓存。

        Args:
            key: 缓存键。
            value: 缓存值。
        """
        expire_at = time.time() + self._ttl if self._ttl > 0 else 0.0

        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = (value, expire_at)
            return

        # 容量溢出时驱逐最久未使用的条目
        while len(self._data) >= self._max_size:
            self._evict_one()

        self._data[key] = (value, expire_at)

    def remove(self, key: Hashable) -> bool:
        """移除指定键。

        Args:
            key: 缓存键。

        Returns:
            是否成功移除。
        """
        if key in self._data:
            del self._data[key]
            return True
        return False

    def contains(self, key: Hashable) -> bool:
        """检查键是否存在且未过期。

        Args:
            key: 缓存键。

        Returns:
            是否存在。
        """
        if key not in self._data:
            return False

        _, expire_at = self._data[key]
        if self._ttl > 0 and time.time() > expire_at:
            del self._data[key]
            self._stats.expirations += 1
            return False

        return True

    def clear(self) -> None:
        """清空所有缓存条目。"""
        self._data.clear()

    def size(self) -> int:
        """当前缓存条目数。"""
        return len(self._data)

    def keys(self) -> list[Hashable]:
        """返回所有缓存键。"""
        return list(self._data.keys())

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        """驱逐最久未使用的条目。"""
        if not self._data:
            return
        # OrderedDict 的第一个元素即为最久未使用
        self._data.popitem(last=False)
        self._stats.evictions += 1

    def purge_expired(self) -> int:
        """清理所有过期条目。

        Returns:
            清理的条目数。
        """
        now = time.time()
        expired_keys = [
            k
            for k, (_, exp) in self._data.items()
            if self._ttl > 0 and now > exp
        ]
        for k in expired_keys:
            del self._data[k]
        self._stats.expirations += len(expired_keys)
        return len(expired_keys)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: Hashable) -> bool:
        return self.contains(key)

    def __repr__(self) -> str:
        return (
            f"LRUCache(max_size={self._max_size}, ttl={self._ttl}, "
            f"size={len(self._data)}, hit_rate={self._stats.hit_rate:.2%})"
        )
