"""因子注册表 FactorRegistry 和 依赖图 FactorDependencyGraph 测试套件。

覆盖:
- 因子注册 / 注销
- 因子查询 (按分类、标签、关键词)
- 因子计算
- 依赖添加 / 移除
- 拓扑排序
- 循环检测
- 边界条件
"""

import time

import numpy as np
import pytest

from smartalpha.registry.factor import FactorRegistry, FactorMetadata
from smartalpha.registry.dependency import (
    FactorDependencyGraph,
    DependencyCycleError,
)


# ============================================================================
# FactorRegistry 测试
# ============================================================================


class TestFactorRegistryRegistration:
    """因子注册与注销测试。"""

    def test_register_factor(self, registry):
        """注册因子。"""
        meta = FactorMetadata(
            name="momentum",
            expression="RANK(DELTA($close, 5))",
            category="trend",
        )
        registry.register(meta)
        assert registry.exists("momentum")
        assert registry.count() == 1

    def test_register_with_computor(self, registry):
        """注册因子并绑定计算函数。"""
        meta = FactorMetadata(name="double", expression="$close * 2")

        def computor(ctx):
            return ctx.get("close", 0) * 2

        registry.register(meta, computor)
        assert registry.has_computor("double")
        result = registry.compute("double", {"close": 10.0})
        assert result == 20.0

    def test_register_duplicate_updates(self, registry):
        """重复注册应更新而非报错。"""
        meta1 = FactorMetadata(name="factor1", expression="RANK($close)")
        registry.register(meta1)
        meta2 = FactorMetadata(name="factor1", expression="MEAN($close, 20)")
        registry.register(meta2)
        assert registry.count() == 1
        factor = registry.get("factor1")
        assert factor.expression == "MEAN($close, 20)"

    def test_unregister(self, registry):
        """注销因子。"""
        registry.register(FactorMetadata(name="temp", expression="$close"))
        assert registry.unregister("temp") is True
        assert not registry.exists("temp")

    def test_unregister_nonexistent(self, registry):
        """注销不存在的因子返回 False。"""
        assert registry.unregister("nonexistent") is False

    def test_case_insensitive_names(self, registry):
        """因子名大小写不敏感。"""
        registry.register(FactorMetadata(name="Momentum", expression="$close"))
        assert registry.exists("momentum")
        assert registry.exists("MOMENTUM")
        assert registry.get("momentum") is not None

    def test_bind_computor(self, registry):
        """为已有因子绑定计算函数。"""
        registry.register(FactorMetadata(name="factor", expression="$close"))
        registry.bind_computor("factor", lambda ctx: 42)
        assert registry.has_computor("factor")
        assert registry.compute("factor") == 42

    def test_bind_computor_nonexistent(self, registry):
        """为不存在的因子绑定应抛出 KeyError。"""
        with pytest.raises(KeyError):
            registry.bind_computor("nonexistent", lambda ctx: None)


class TestFactorRegistryQuery:
    """因子查询测试。"""

    def test_get_factor(self, registry):
        """获取因子元数据。"""
        meta = FactorMetadata(
            name="rsi_14",
            display_name="14日RSI",
            expression="RSI($close, 14)",
            category="technical",
            tags=["trend", "oscillator"],
        )
        registry.register(meta)
        factor = registry.get("rsi_14")
        assert factor is not None
        assert factor.display_name == "14日RSI"
        assert factor.category == "technical"
        assert factor.tags == ["trend", "oscillator"]

    def test_get_nonexistent(self, registry):
        """获取不存在的因子返回 None。"""
        assert registry.get("nonexistent") is None

    def test_list_all(self, registry):
        """列出所有因子。"""
        registry.register(FactorMetadata(name="f1", expression="$close"))
        registry.register(FactorMetadata(name="f2", expression="$open"))
        all_factors = registry.list_all()
        assert len(all_factors) == 2

    def test_list_names(self, registry):
        """列出所有因子名。"""
        registry.register(FactorMetadata(name="f2", expression="$close"))
        registry.register(FactorMetadata(name="f1", expression="$open"))
        names = registry.list_names()
        assert names == ["F1", "F2"]

    def test_find_by_category(self, registry):
        """按分类查找。"""
        registry.register(FactorMetadata(name="f1", expression="$close", category="tech"))
        registry.register(FactorMetadata(name="f2", expression="$open", category="tech"))
        registry.register(FactorMetadata(name="f3", expression="$high", category="basic"))
        tech_factors = registry.find_by_category("tech")
        assert len(tech_factors) == 2

    def test_find_by_tag(self, registry):
        """按标签查找。"""
        registry.register(FactorMetadata(name="f1", expression="$close", tags=["alpha", "trend"]))
        registry.register(FactorMetadata(name="f2", expression="$open", tags=["alpha"]))
        registry.register(FactorMetadata(name="f3", expression="$high", tags=["beta"]))
        alpha_factors = registry.find_by_tag("alpha")
        assert len(alpha_factors) == 2

    def test_search(self, registry):
        """关键词搜索。"""
        registry.register(FactorMetadata(
            name="momentum",
            display_name="动量因子",
            expression="DELTA($close, 5)",
            description="衡量价格变化率",
            tags=["trend"],
        ))
        results = registry.search("动量")
        assert len(results) == 1
        results = registry.search("close")
        assert len(results) == 1

    def test_categories(self, registry):
        """获取所有分类。"""
        registry.register(FactorMetadata(name="f1", expression="$close", category="tech"))
        registry.register(FactorMetadata(name="f2", expression="$open", category="basic"))
        cats = registry.categories()
        assert cats == {"tech", "basic"}

    def test_tags(self, registry):
        """获取所有标签。"""
        registry.register(FactorMetadata(name="f1", expression="$close", tags=["a", "b"]))
        registry.register(FactorMetadata(name="f2", expression="$open", tags=["b", "c"]))
        all_tags = registry.tags()
        assert all_tags == {"a", "b", "c"}

    def test_count(self, registry):
        """因子计数。"""
        assert registry.count() == 0
        registry.register(FactorMetadata(name="f1", expression="$close"))
        assert registry.count() == 1

    def test_contains(self, registry):
        """__contains__ 方法。"""
        registry.register(FactorMetadata(name="f1", expression="$close"))
        assert "f1" in registry
        assert "f2" not in registry

    def test_factor_metadata_to_dict(self, registry):
        """FactorMetadata 转字典。"""
        meta = FactorMetadata(
            name="test",
            display_name="测试因子",
            expression="$close",
            category="test",
            description="描述",
            tags=["tag1"],
        )
        d = meta.to_dict()
        assert d["name"] == "test"
        assert d["display_name"] == "测试因子"
        assert d["expression"] == "$close"
        assert d["category"] == "test"
        assert d["tags"] == ["tag1"]


class TestFactorRegistryCompute:
    """因子计算测试。"""

    def test_compute_with_context(self, registry):
        """带上下文计算。"""
        meta = FactorMetadata(name="double", expression="$x * 2")
        registry.register(meta, lambda ctx: ctx["x"] * 2)
        result = registry.compute("double", {"x": 21})
        assert result == 42

    def test_compute_no_computor(self, registry):
        """无计算函数应抛出 KeyError。"""
        registry.register(FactorMetadata(name="f1", expression="$close"))
        with pytest.raises(KeyError):
            registry.compute("f1")

    def test_compute_nonexistent(self, registry):
        """计算不存在的因子应抛出 KeyError。"""
        with pytest.raises(KeyError):
            registry.compute("nonexistent")


# ============================================================================
# FactorDependencyGraph 测试
# ============================================================================


class TestDependencyGraphConstruction:
    """依赖图构建测试。"""

    def test_add_node(self, dep_graph):
        """添加节点。"""
        dep_graph.add_node("factor_a")
        assert "FACTOR_A" in dep_graph.nodes()
        assert dep_graph.node_count() == 1

    def test_add_dependency(self, dep_graph):
        """添加依赖关系。"""
        dep_graph.add_dependency("factor_a", ["close", "open"])
        deps = dep_graph.get_dependencies("factor_a")
        assert deps == {"CLOSE", "OPEN"}

    def test_add_dependency_edge(self, dep_graph):
        """添加单条依赖边。"""
        dep_graph.add_dependency_edge("factor_a", "close")
        deps = dep_graph.get_dependencies("factor_a")
        assert deps == {"CLOSE"}

    def test_remove_dependency(self, dep_graph):
        """移除依赖。"""
        dep_graph.add_dependency("factor_a", ["close", "open"])
        assert dep_graph.remove_dependency("factor_a", "close") is True
        deps = dep_graph.get_dependencies("factor_a")
        assert deps == {"OPEN"}

    def test_remove_dependency_nonexistent(self, dep_graph):
        """移除不存在的依赖返回 False。"""
        assert dep_graph.remove_dependency("a", "b") is False

    def test_remove_node(self, dep_graph):
        """移除节点。"""
        dep_graph.add_dependency("factor_a", ["close"])
        assert dep_graph.remove_node("factor_a") is True
        assert "FACTOR_A" not in dep_graph.nodes()
        assert dep_graph.node_count() == 1

    def test_remove_node_nonexistent(self, dep_graph):
        """移除不存在的节点返回 False。"""
        assert dep_graph.remove_node("nonexistent") is False


class TestDependencyGraphQuery:
    """依赖图查询测试。"""

    def test_get_dependencies(self, dep_graph):
        """获取直接依赖。"""
        dep_graph.add_dependency("a", ["b", "c"])
        assert dep_graph.get_dependencies("a") == {"B", "C"}

    def test_get_dependents(self, dep_graph):
        """获取反向依赖。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("c", ["b"])
        dependents = dep_graph.get_dependents("b")
        assert dependents == {"A", "C"}

    def test_get_all_dependencies(self, dep_graph):
        """获取所有传递依赖。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("b", ["c"])
        all_deps = dep_graph.get_all_dependencies("a")
        assert "B" in all_deps
        assert "C" in all_deps

    def test_get_all_dependents(self, dep_graph):
        """获取所有传递反向依赖。"""
        dep_graph.add_dependency("b", ["a"])
        dep_graph.add_dependency("c", ["b"])
        all_dependents = dep_graph.get_all_dependents("a")
        assert "B" in all_dependents
        assert "C" in all_dependents

    def test_node_count(self, dep_graph):
        """节点计数。"""
        dep_graph.add_node("a")
        dep_graph.add_node("b")
        assert dep_graph.node_count() == 2

    def test_edge_count(self, dep_graph):
        """边计数。"""
        dep_graph.add_dependency("a", ["b", "c"])
        dep_graph.add_dependency("d", ["b"])
        assert dep_graph.edge_count() == 3

    def test_nodes(self, dep_graph):
        """获取所有节点。"""
        dep_graph.add_node("a")
        dep_graph.add_node("b")
        nodes = dep_graph.nodes()
        assert nodes == {"A", "B"}

    def test_get_graph_data(self, dep_graph):
        """导出图数据。"""
        dep_graph.add_dependency("a", ["b", "c"])
        data = dep_graph.get_graph_data()
        assert data == {"A": ["B", "C"], "B": [], "C": []}

    def test_dependencies_case_insensitive(self, dep_graph):
        """依赖关系大小写不敏感。"""
        dep_graph.add_dependency("Factor_A", ["Close"])
        deps = dep_graph.get_dependencies("factor_a")
        assert deps == {"CLOSE"}


class TestTopologicalSort:
    """拓扑排序测试。"""

    def test_simple_sort(self, dep_graph):
        """简单拓扑排序。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("b", ["c"])
        order = dep_graph.topological_sort()
        assert order.index("C") < order.index("B")
        assert order.index("B") < order.index("A")

    def test_diamond_dependency(self, dep_graph):
        """菱形依赖排序。"""
        dep_graph.add_dependency("a", ["b", "c"])
        dep_graph.add_dependency("b", ["d"])
        dep_graph.add_dependency("c", ["d"])
        order = dep_graph.topological_sort()
        assert order.index("D") < order.index("B")
        assert order.index("D") < order.index("C")
        assert order.index("B") < order.index("A")
        assert order.index("C") < order.index("A")

    def test_independent_nodes(self, dep_graph):
        """独立节点排序。"""
        dep_graph.add_node("a")
        dep_graph.add_node("b")
        dep_graph.add_node("c")
        order = dep_graph.topological_sort()
        assert len(order) == 3

    def test_single_node(self, dep_graph):
        """单节点。"""
        dep_graph.add_node("solo")
        order = dep_graph.topological_sort()
        assert order == ["SOLO"]

    def test_sort_deterministic(self, dep_graph):
        """排序确定性 (同依赖节点字母序)。"""
        dep_graph.add_dependency("c", ["base"])
        dep_graph.add_dependency("a", ["base"])
        dep_graph.add_dependency("b", ["base"])
        order = dep_graph.topological_sort()
        # 同层节点应字母序排列
        base_idx = order.index("BASE")
        for n in ["A", "B", "C"]:
            assert order.index(n) > base_idx


class TestCycleDetection:
    """循环依赖检测测试。"""

    def test_no_cycle(self, dep_graph):
        """无循环依赖。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("b", ["c"])
        assert not dep_graph.has_cycle()

    def test_simple_cycle(self, dep_graph):
        """简单循环。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("b", ["a"])
        assert dep_graph.has_cycle()

    def test_three_way_cycle(self, dep_graph):
        """三方循环。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("b", ["c"])
        dep_graph.add_dependency("c", ["a"])
        assert dep_graph.has_cycle()

    def test_self_loop(self, dep_graph):
        """自循环。"""
        dep_graph.add_dependency("a", ["a"])
        assert dep_graph.has_cycle()

    def test_cycle_raises_error(self, dep_graph):
        """循环依赖在拓扑排序时抛出异常。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("b", ["a"])
        with pytest.raises(DependencyCycleError):
            dep_graph.topological_sort()

    def test_cycle_error_contains_path(self, dep_graph):
        """循环异常包含循环路径。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("b", ["a"])
        try:
            dep_graph.topological_sort()
            assert False
        except DependencyCycleError as e:
            assert len(e.cycle) >= 2

    def test_dependency_error_str(self, dep_graph):
        """异常消息包含循环信息。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("b", ["a"])
        try:
            dep_graph.topological_sort()
        except DependencyCycleError as e:
            msg = str(e)
            assert "循环" in msg or "cycle" in msg.lower()

    def test_no_cycle_after_removal(self, dep_graph):
        """移除循环后应无循环。"""
        dep_graph.add_dependency("a", ["b"])
        dep_graph.add_dependency("b", ["a"])
        assert dep_graph.has_cycle()
        dep_graph.remove_dependency("a", "b")
        assert not dep_graph.has_cycle()


class TestEdgeCases:
    """边界条件测试。"""

    def test_empty_registry(self, registry):
        """空注册表。"""
        assert registry.count() == 0
        assert registry.list_all() == []
        assert registry.list_names() == []
        assert registry.categories() == set()

    def test_empty_graph(self, dep_graph):
        """空依赖图。"""
        assert dep_graph.node_count() == 0
        assert dep_graph.edge_count() == 0
        assert dep_graph.topological_sort() == []
        assert not dep_graph.has_cycle()

    def test_registry_repr(self, registry):
        """注册表 repr。"""
        r = repr(registry)
        assert "FactorRegistry" in r

    def test_graph_repr(self, dep_graph):
        """依赖图 repr。"""
        r = repr(dep_graph)
        assert "FactorDependencyGraph" in r

    def test_factor_metadata_defaults(self):
        """FactorMetadata 默认值。"""
        meta = FactorMetadata(name="test")
        assert meta.display_name == ""
        assert meta.category == "未分类"
        assert meta.expression == ""
        assert meta.description == ""
        assert meta.tags == []
        assert meta.owner == ""

    def test_metadata_timestamps(self):
        """元数据时间戳。"""
        before = time.time()
        meta = FactorMetadata(name="test")
        after = time.time()
        assert before <= meta.created_at <= after
        assert before <= meta.updated_at <= after

    def test_registry_update_preserves_created_at(self, registry):
        """更新注册时保留原创建时间。"""
        meta1 = FactorMetadata(name="factor", expression="v1")
        registry.register(meta1)
        original_created = registry.get("factor").created_at

        meta2 = FactorMetadata(name="factor", expression="v2")
        time.sleep(0.01)
        registry.register(meta2)
        updated = registry.get("factor")
        assert updated.created_at == original_created
        assert updated.updated_at >= original_created