"""因子依赖图模块。

构建和分析因子之间的依赖关系，提供：
- 依赖图的构建与维护
- 拓扑排序（确定因子计算顺序）
- 循环依赖检测
- 依赖链追踪
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


class DependencyCycleError(Exception):
    """循环依赖异常。"""

    def __init__(self, cycle: List[str]) -> None:
        self.cycle = cycle
        super().__init__(f"检测到循环依赖: {' -> '.join(cycle)}")


class FactorDependencyGraph:
    """因子依赖图。

    维护因子之间的有向依赖关系，支持拓扑排序与循环检测。

    使用示例::

        graph = FactorDependencyGraph()
        graph.add_dependency("momentum", ["close", "volume"])
        graph.add_dependency("alpha", ["momentum", "beta"])
        order = graph.topological_sort()  # ['close', 'volume', 'momentum', 'beta', 'alpha']
        has_cycle = graph.has_cycle()
    """

    def __init__(self) -> None:
        # 依赖关系: factor -> set of dependencies
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        # 反向依赖: dependency -> set of factors that depend on it
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        # 所有节点
        self._nodes: Set[str] = set()

    # ------------------------------------------------------------------
    # 图构建
    # ------------------------------------------------------------------

    def add_node(self, name: str) -> None:
        """添加节点（无依赖）。

        Args:
            name: 因子名。
        """
        self._nodes.add(name.upper())

    def add_dependency(
        self, factor: str, dependencies: List[str]
    ) -> None:
        """添加因子依赖关系。

        Args:
            factor: 因子名。
            dependencies: 该因子依赖的其他因子名列表。
        """
        factor_upper = factor.upper()
        self._nodes.add(factor_upper)

        for dep in dependencies:
            dep_upper = dep.upper()
            self._nodes.add(dep_upper)
            self._dependencies[factor_upper].add(dep_upper)
            self._dependents[dep_upper].add(factor_upper)

    def add_dependency_edge(self, factor: str, depends_on: str) -> None:
        """添加单条依赖边。

        Args:
            factor: 因子名。
            depends_on: 依赖的因子名。
        """
        self.add_dependency(factor, [depends_on])

    def remove_dependency(self, factor: str, dependency: str) -> bool:
        """移除依赖关系。

        Args:
            factor: 因子名。
            dependency: 要移除的依赖。

        Returns:
            是否存在并移除成功。
        """
        factor_upper = factor.upper()
        dep_upper = dependency.upper()

        removed_from_deps = dep_upper in self._dependencies[factor_upper]
        self._dependencies[factor_upper].discard(dep_upper)
        removed_from_dependents = factor_upper in self._dependents[dep_upper]
        self._dependents[dep_upper].discard(factor_upper)
        return removed_from_deps or removed_from_dependents

    def remove_node(self, name: str) -> bool:
        """移除节点及其所有关联依赖。

        Args:
            name: 因子名。

        Returns:
            是否存在并移除成功。
        """
        key = name.upper()
        if key not in self._nodes:
            return False

        # 移除所有从该节点发出的依赖
        for dep in list(self._dependencies[key]):
            self._dependents[dep].discard(key)
        self._dependencies.pop(key, None)

        # 移除所有指向该节点的依赖
        for dependent in list(self._dependents[key]):
            self._dependencies[dependent].discard(key)
        self._dependents.pop(key, None)

        self._nodes.discard(key)
        return True

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get_dependencies(self, name: str) -> Set[str]:
        """获取指定因子的直接依赖集合。

        Args:
            name: 因子名。

        Returns:
            直接依赖集合。
        """
        return self._dependencies.get(name.upper(), set()).copy()

    def get_dependents(self, name: str) -> Set[str]:
        """获取依赖于指定因子的所有因子。

        Args:
            name: 因子名。

        Returns:
            反向依赖集合。
        """
        return self._dependents.get(name.upper(), set()).copy()

    def get_all_dependencies(self, name: str) -> Set[str]:
        """递归获取所有传递依赖。

        Args:
            name: 因子名。

        Returns:
            所有传递依赖的集合（不包括自身）。
        """
        visited: Set[str] = set()
        self._collect_deps_recursive(name.upper(), visited)
        visited.discard(name.upper())
        return visited

    def get_all_dependents(self, name: str) -> Set[str]:
        """递归获取所有传递反向依赖。

        Args:
            name: 因子名。

        Returns:
            所有传递反向依赖的集合（不包括自身）。
        """
        visited: Set[str] = set()
        self._collect_dependents_recursive(name.upper(), visited)
        visited.discard(name.upper())
        return visited

    def nodes(self) -> Set[str]:
        """获取所有节点。"""
        return self._nodes.copy()

    def node_count(self) -> int:
        """获取节点数量。"""
        return len(self._nodes)

    def edge_count(self) -> int:
        """获取依赖边数量。"""
        return sum(len(deps) for deps in self._dependencies.values())

    def get_graph_data(self) -> Dict[str, List[str]]:
        """导出图数据为 {factor: [deps]} 格式。

        Returns:
            图数据字典。
        """
        return {
            node: sorted(self._dependencies.get(node, set()))
            for node in sorted(self._nodes)
        }

    # ------------------------------------------------------------------
    # 拓扑排序与循环检测
    # ------------------------------------------------------------------

    def topological_sort(self) -> List[str]:
        """执行拓扑排序，返回因子的计算顺序。

        排序保证：若因子 A 依赖因子 B，则 B 排在 A 之前。

        Returns:
            拓扑排序后的因子列表。

        Raises:
            DependencyCycleError: 存在循环依赖时。
        """
        if self.has_cycle():
            cycle = self._find_cycle()
            raise DependencyCycleError(cycle)

        # Kahn's 算法
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for dep in self._dependencies.get(node, set()):
                if dep in in_degree:
                    pass  # 确保 dep 在图中
                # node 依赖 dep，所以 dep 需先计算
                # 入度: node 依赖 dep 意味着 dep -> node 的边
                # 为了拓扑排序，我们需要反向图：依赖者有入度

        # 构建入度表（factor 的入度 = 它依赖的数量）
        in_deg: Dict[str, int] = {
            n: len(self._dependencies.get(n, set())) for n in self._nodes
        }

        # 构建正向邻接表：dep -> factors that depend on dep
        adj: Dict[str, Set[str]] = defaultdict(set)
        for node in self._nodes:
            for dep in self._dependencies.get(node, set()):
                adj[dep].add(node)

        # Kahn's 算法
        queue: deque[str] = deque(
            sorted(n for n in self._nodes if in_deg[n] == 0)
        )
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for dependent in sorted(adj.get(node, set())):
                in_deg[dependent] -= 1
                if in_deg[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._nodes):
            cycle = self._find_cycle()
            raise DependencyCycleError(cycle)

        return result

    def has_cycle(self) -> bool:
        """检测是否存在循环依赖。

        Returns:
            是否存在循环。
        """
        # 使用三色 DFS
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in self._nodes}

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for dep in self._dependencies.get(node, set()):
                if color[dep] == GRAY:
                    return True  # 发现回边
                if color[dep] == WHITE and dfs(dep):
                    return True
            color[node] = BLACK
            return False

        for node in self._nodes:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        return False

    def _find_cycle(self) -> List[str]:
        """找到一个循环依赖的路径。

        Returns:
            循环路径列表，若不存在循环返回空列表。
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in self._nodes}
        parent: Dict[str, Optional[str]] = {n: None for n in self._nodes}
        cycle_path: List[str] = []

        def dfs(node: str) -> bool:
            color[node] = GRAY
            for dep in self._dependencies.get(node, set()):
                if color[dep] == GRAY:
                    # 找到循环，回溯路径
                    path = [dep, node]
                    p = parent[node]
                    while p is not None and p != dep:
                        path.append(p)
                        p = parent[p]
                    path.append(dep)
                    cycle_path.extend(reversed(path))
                    return True
                if color[dep] == WHITE:
                    parent[dep] = node
                    if dfs(dep):
                        return True
            color[node] = BLACK
            return False

        for node in self._nodes:
            if color[node] == WHITE:
                if dfs(node):
                    break

        return cycle_path

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _collect_deps_recursive(
        self, node: str, visited: Set[str]
    ) -> None:
        """递归收集所有传递依赖。"""
        if node in visited:
            return
        visited.add(node)
        for dep in self._dependencies.get(node, set()):
            self._collect_deps_recursive(dep, visited)

    def _collect_dependents_recursive(
        self, node: str, visited: Set[str]
    ) -> None:
        """递归收集所有传递反向依赖。"""
        if node in visited:
            return
        visited.add(node)
        for dependent in self._dependents.get(node, set()):
            self._collect_dependents_recursive(dependent, visited)

    def __repr__(self) -> str:
        return (
            f"FactorDependencyGraph(nodes={len(self._nodes)}, "
            f"edges={self.edge_count()})"
        )
