"""ChromaDB 记忆存储测试。

验证 ChromaMemoryStore 的核心功能：
- store_decision / query_similar 往返一致性
- store_market_snapshot 向量存储
- ChromaDB 不可用时的降级行为
"""
import pytest
import numpy as np
from smartalpha.memory.chroma_store import ChromaMemoryStore


class TestChromaMemoryStore:
    """ChromaMemoryStore 功能测试。"""

    @pytest.fixture(scope="class")
    def store(self):
        """创建临时目录的 store 实例（class 级别，只初始化一次）。"""
        import tempfile
        tmp = tempfile.mkdtemp(prefix="smartalpha_test_")
        s = ChromaMemoryStore(persist_dir=tmp)
        yield s
        import shutil
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def test_init_creates_store(self, store):
        """初始化应创建 store 实例并设置 _available 状态。"""
        assert store is not None
        assert hasattr(store, '_available')
        assert isinstance(store._available, bool)

    def test_persist_dir_set(self, store):
        """persist_dir 应被正确设置。"""
        assert store._persist_dir is not None
        assert "smartalpha_test_" in store._persist_dir

    # ------------------------------------------------------------------
    # 决策存储与检索
    # ------------------------------------------------------------------

    def test_store_decision_returns_true(self, store):
        """store_decision 在 ChromaDB 可用时应返回 True。"""
        ok = store.store_decision(
            "000001.SZ",
            "fundamental",
            {"signal": "bullish", "confidence": 0.8, "reasoning": "财务数据良好"},
            {"date": "2024-07-01"},
        )
        if store._available:
            assert ok is True
        else:
            assert ok is False  # 降级模式

    def test_store_and_query_roundtrip(self, store):
        """存储后查询应能检索到相关内容。"""
        if not store._available:
            pytest.skip("ChromaDB 不可用，跳过往返测试")

        store.store_decision(
            "000001.SZ",
            "fundamental",
            {"signal": "bullish", "confidence": 0.8, "reasoning": "低估值的银行股"},
            {"date": "2024-07-01"},
        )
        result = store.query_similar("agent_decisions", "银行股估值", k=3)
        assert len(result) > 0

    def test_store_multiple_decisions(self, store):
        """多个决策存储不应报错。"""
        for i, agent in enumerate(["fundamental", "technical", "sentiment"]):
            ok = store.store_decision(
                f"00000{i+1}.SZ",
                agent,
                {"signal": "neutral", "confidence": 0.5, "reasoning": "test"},
                {"date": f"2024-07-0{i+1}"},
            )
            if store._available:
                assert ok is True

    def test_store_decision_without_metadata(self, store):
        """metadata 应为可选参数。"""
        ok = store.store_decision(
            "000001.SZ",
            "sentiment",
            {"signal": "bearish", "confidence": 0.3, "reasoning": "高波动"},
        )
        if store._available:
            assert ok is True

    # ------------------------------------------------------------------
    # 市场快照
    # ------------------------------------------------------------------

    def test_store_market_snapshot(self, store):
        """store_market_snapshot 应正常存储特征向量。"""
        features = np.random.randn(10).astype(np.float32)
        ok = store.store_market_snapshot(
            "000001.SZ",
            features,
            {"date": "2024-07-01"},
        )
        if store._available:
            assert ok is True
        else:
            assert ok is False

    def test_query_market_patterns(self, store):
        """query_similar 对 market_patterns collection 应返回结果。"""
        if not store._available:
            pytest.skip("ChromaDB 不可用")

        features = np.random.randn(10).astype(np.float32)
        store.store_market_snapshot("000001.SZ", features)
        result = store.query_similar("market_patterns", "000001.SZ 银行", k=3)
        assert isinstance(result, str)

    def test_market_snapshot_scalar_features(self, store):
        """标量特征应被转换为列表。"""
        features = np.float32(3.14)
        ok = store.store_market_snapshot("000001.SZ", features)
        if store._available:
            assert ok is True

    # ------------------------------------------------------------------
    # 内存统计
    # ------------------------------------------------------------------

    def test_count_returns_int(self, store):
        """count 应返回整数。"""
        c = store.count("agent_decisions")
        assert isinstance(c, int)
        assert c >= 0

    def test_stats_returns_dict(self, store):
        """stats 应返回字典。"""
        s = store.stats()
        assert isinstance(s, dict)

    # ------------------------------------------------------------------
    # 容错降级
    # ------------------------------------------------------------------

    def test_query_unavailable_collection(self, store):
        """查询不存在的 collection 应返回空字符串。"""
        result = store.query_similar("nonexistent_collection", "test", k=3)
        assert result == ""

    def test_query_by_stock_returns_list(self, store):
        """query_by_stock 应返回列表。"""
        results = store.query_by_stock("000001.SZ", k=3)
        assert isinstance(results, list)

    def test_store_reflection(self, store):
        """store_reflection 不应抛异常。"""
        ok = store.store_reflection(
            "000001.SZ",
            "fundamental",
            "success",
            "预测准确，信号方向正确",
        )
        if store._available:
            assert ok is True
        else:
            assert ok is False
