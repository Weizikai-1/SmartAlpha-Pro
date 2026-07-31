"""LRU 缓存 LRUCache 测试套件。

覆盖:
- put/get 基本操作
- 命中率统计
- 淘汰机制
- TTL 过期
- 边界条件
"""

import time

import pytest

from smartalpha.storage.cache import LRUCache, CacheStats


class TestBasicOperations:
    """基本操作测试。"""

    def test_put_and_get(self, cache):
        """放入和获取。"""
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_miss(self, cache):
        """获取不存在的键返回 None。"""
        assert cache.get("nonexistent") is None

    def test_update_existing(self, cache):
        """更新已有键。"""
        cache.put("key1", "value1")
        cache.put("key1", "value2")
        assert cache.get("key1") == "value2"
        assert cache.size() == 1

    def test_multiple_keys(self, cache):
        """多个键值对。"""
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        assert cache.get("a") == 1
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.size() == 3

    def test_remove(self, cache):
        """移除指定键。"""
        cache.put("key1", "value1")
        assert cache.remove("key1") is True
        assert cache.get("key1") is None
        assert cache.remove("nonexistent") is False

    def test_contains(self, cache):
        """检查键是否存在。"""
        cache.put("key1", "value1")
        assert cache.contains("key1") is True
        assert cache.contains("nonexistent") is False

    def test_clear(self, cache):
        """清空缓存。"""
        cache.put("a", 1)
        cache.put("b", 2)
        cache.clear()
        assert cache.size() == 0
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_size(self, cache):
        """缓存大小。"""
        assert cache.size() == 0
        cache.put("a", 1)
        assert cache.size() == 1
        cache.put("b", 2)
        assert cache.size() == 2

    def test_keys(self, cache):
        """获取所有键。"""
        cache.put("a", 1)
        cache.put("b", 2)
        keys = cache.keys()
        assert "a" in keys
        assert "b" in keys
        assert len(keys) == 2

    def test_len(self, cache):
        """__len__ 方法。"""
        assert len(cache) == 0
        cache.put("a", 1)
        assert len(cache) == 1

    def test_contains_dunder(self, cache):
        """__contains__ 方法。"""
        cache.put("a", 1)
        assert "a" in cache
        assert "b" not in cache

    def test_repr(self, cache):
        """__repr__ 方法。"""
        cache.put("a", 1)
        r = repr(cache)
        assert "LRUCache" in r
        assert "size=1" in r


class TestStats:
    """统计信息测试。"""

    def test_hit_miss_counts(self, cache):
        """命中/未命中计数。"""
        cache.put("key1", "value1")
        cache.get("key1")
        cache.get("key1")
        cache.get("nonexistent")
        assert cache.stats.hits == 2
        assert cache.stats.misses == 1

    def test_hit_rate(self, cache):
        """命中率计算。"""
        cache.put("key1", "value1")
        cache.get("key1")
        cache.get("key1")
        cache.get("nonexistent")
        assert abs(cache.stats.hit_rate - 2.0 / 3.0) < 1e-10

    def test_hit_rate_zero_access(self, cache):
        """零访问时命中率为 0。"""
        assert cache.stats.hit_rate == 0.0

    def test_total(self, cache):
        """总访问次数。"""
        cache.get("a")
        cache.get("b")
        cache.get("c")
        assert cache.stats.total == 3

    def test_stats_reset(self, cache):
        """统计重置。"""
        cache.put("a", 1)
        cache.get("a")
        cache.get("b")
        cache.stats.reset()
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0
        assert cache.stats.total == 0

    def test_stats_after_eviction(self, cache):
        """淘汰后统计。"""
        small_cache = LRUCache(max_size=3)
        small_cache.put("a", 1)
        small_cache.put("b", 2)
        small_cache.put("c", 3)
        small_cache.put("d", 4)
        assert small_cache.stats.evictions == 1
        assert small_cache.size() == 3


class TestEviction:
    """淘汰机制测试。"""

    def test_eviction_removes_least_recently_used(self):
        """淘汰最久未使用的条目。"""
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_access_moves_to_end(self):
        """访问条目应移至最近使用位置。"""
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")
        cache.put("d", 4)
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3
        assert cache.get("d") == 4

    def test_update_does_not_increase_size(self):
        """更新不增加大小。"""
        cache = LRUCache(max_size=3)
        cache.put("a", 1)
        cache.put("a", 2)
        assert cache.size() == 1

    def test_max_size_one(self):
        """max_size 为 1。"""
        cache = LRUCache(max_size=1)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.size() == 1
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_max_size_minimum(self):
        """构造函数 max_size 为 0 时应至少为 1。"""
        cache = LRUCache(max_size=0)
        assert cache.max_size == 1


class TestTTL:
    """TTL 过期测试。"""

    def test_ttl_expiry(self):
        """TTL 过期后返回 None。"""
        cache = LRUCache(max_size=256, ttl=0.1)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_ttl_no_expiry(self):
        """TTL 为 0 永不过期。"""
        cache = LRUCache(max_size=256, ttl=0)
        cache.put("key1", "value1")
        time.sleep(0.05)
        assert cache.get("key1") == "value1"

    def test_ttl_expiration_count(self):
        """过期计数。"""
        cache = LRUCache(max_size=256, ttl=0.1)
        cache.put("a", 1)
        cache.put("b", 2)
        time.sleep(0.15)
        cache.get("a")
        assert cache.stats.expirations == 1

    def test_purge_expired(self):
        """主动清理过期条目。"""
        cache = LRUCache(max_size=256, ttl=0.1)
        cache.put("a", 1)
        cache.put("b", 2)
        time.sleep(0.15)
        removed = cache.purge_expired()
        assert removed == 2
        assert cache.size() == 0

    def test_ttl_contains_check(self):
        """contains 对过期条目返回 False。"""
        cache = LRUCache(max_size=256, ttl=0.1)
        cache.put("key1", "value1")
        time.sleep(0.15)
        assert cache.contains("key1") is False


class TestEdgeCases:
    """边界条件测试。"""

    def test_none_value(self, cache):
        """None 值。"""
        cache.put("key", None)
        assert cache.get("key") is None
        assert cache.contains("key")

    def test_complex_values(self, cache):
        """复杂对象作为值。"""
        data = [1, 2, {"key": "value"}]
        cache.put("complex", data)
        assert cache.get("complex") == data

    def test_tuple_key(self, cache):
        """元组键。"""
        cache.put(("a", "b"), "value")
        assert cache.get(("a", "b")) == "value"

    def test_large_number_of_entries(self):
        """大量条目。"""
        cache = LRUCache(max_size=10000)
        for i in range(5000):
            cache.put(f"key_{i}", i)
        assert cache.size() == 5000
        assert cache.get("key_0") == 0
        assert cache.get("key_4999") == 4999

    def test_concurrent_like_access(self, cache):
        """模拟交替访问。"""
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        for _ in range(100):
            cache.get("a")
            cache.get("b")
            cache.get("c")
        assert cache.stats.hits == 300

    def test_max_size_property(self, cache):
        """max_size 属性。"""
        assert cache.max_size == 256

    def test_ttl_property(self, cache):
        """ttl 属性。"""
        assert cache.ttl == 0.0

    def test_cache_stats_dataclass(self):
        """CacheStats 数据类。"""
        stats = CacheStats(hits=10, misses=5)
        assert stats.hits == 10
        assert stats.misses == 5
        assert stats.total == 15
        assert abs(stats.hit_rate - 10.0 / 15.0) < 1e-10