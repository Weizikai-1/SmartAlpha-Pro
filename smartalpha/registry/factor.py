"""因子注册表模块。

提供因子的注册、查询、分类与元数据管理能力，
是因子库与计算引擎之间的桥梁。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


@dataclass
class FactorMetadata:
    """因子元数据。

    Attributes:
        name: 因子名称（唯一标识）。
        display_name: 显示名称。
        category: 分类（如 "技术指标"、"基础统计"）。
        expression: 因子表达式字符串。
        description: 描述信息。
        tags: 标签集合。
        created_at: 创建时间戳。
        updated_at: 更新时间戳。
        owner: 创建者。
    """

    name: str
    display_name: str = ""
    category: str = "未分类"
    expression: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    owner: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转为字典。"""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "category": self.category,
            "expression": self.expression,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "owner": self.owner,
        }


class FactorRegistry:
    """因子注册表。

    管理因子的元数据与计算函数，支持：
    - 因子注册与注销
    - 按分类/标签/名称查询
    - 因子计算函数绑定
    - 因子依赖关系维护

    使用示例::

        registry = FactorRegistry()
        registry.register(
            FactorMetadata(name="momentum", expression="DELTA($close, 20)", category="趋势")
        )
        meta = registry.get("momentum")
        factors = registry.find_by_category("趋势")
    """

    def __init__(self) -> None:
        self._factors: Dict[str, FactorMetadata] = {}
        self._computors: Dict[str, Callable[..., Any]] = {}

    # ------------------------------------------------------------------
    # 注册 / 注销
    # ------------------------------------------------------------------

    def register(
        self,
        metadata: FactorMetadata,
        computor: Optional[Callable[..., Any]] = None,
    ) -> None:
        """注册因子。

        Args:
            metadata: 因子元数据。
            computor: 可选的计算函数，接受上下文字典作为参数。

        Raises:
            ValueError: 因子名已存在时。
        """
        name = metadata.name.upper()
        if name in self._factors:
            # 更新模式：覆盖元数据但保留原创建时间
            existing = self._factors[name]
            metadata.created_at = existing.created_at
            metadata.updated_at = time.time()
        self._factors[name] = metadata

        if computor is not None:
            self._computors[name] = computor

    def unregister(self, name: str) -> bool:
        """注销因子。

        Args:
            name: 因子名。

        Returns:
            是否成功注销。
        """
        key = name.upper()
        removed = self._factors.pop(key, None) is not None
        self._computors.pop(key, None)
        return removed

    def bind_computor(
        self, name: str, computor: Callable[..., Any]
    ) -> None:
        """为已有因子绑定计算函数。

        Args:
            name: 因子名。
            computor: 计算函数。

        Raises:
            KeyError: 因子不存在时。
        """
        key = name.upper()
        if key not in self._factors:
            raise KeyError(f"因子不存在: {name}")
        self._computors[key] = computor

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[FactorMetadata]:
        """获取因子元数据。

        Args:
            name: 因子名。

        Returns:
            因子元数据，不存在返回 None。
        """
        return self._factors.get(name.upper())

    def exists(self, name: str) -> bool:
        """检查因子是否存在。"""
        return name.upper() in self._factors

    def list_all(self) -> List[FactorMetadata]:
        """列出所有因子。

        Returns:
            因子元数据列表。
        """
        return list(self._factors.values())

    def list_names(self) -> List[str]:
        """列出所有因子名。

        Returns:
            因子名列表。
        """
        return sorted(self._factors.keys())

    def find_by_category(self, category: str) -> List[FactorMetadata]:
        """按分类查找因子。

        Args:
            category: 分类名。

        Returns:
            匹配的因子列表。
        """
        cat_lower = category.lower()
        return [
            m for m in self._factors.values()
            if m.category.lower() == cat_lower
        ]

    def find_by_tag(self, tag: str) -> List[FactorMetadata]:
        """按标签查找因子。

        Args:
            tag: 标签名。

        Returns:
            匹配的因子列表。
        """
        tag_lower = tag.lower()
        return [
            m for m in self._factors.values()
            if tag_lower in [t.lower() for t in m.tags]
        ]

    def search(self, keyword: str) -> List[FactorMetadata]:
        """按关键词搜索因子（匹配名称、显示名、描述、标签）。

        Args:
            keyword: 搜索关键词。

        Returns:
            匹配的因子列表。
        """
        kw = keyword.lower()
        results: List[FactorMetadata] = []
        for m in self._factors.values():
            if (
                kw in m.name.lower()
                or kw in m.display_name.lower()
                or kw in m.description.lower()
                or kw in m.expression.lower()
                or any(kw in t.lower() for t in m.tags)
            ):
                results.append(m)
        return results

    def categories(self) -> Set[str]:
        """获取所有分类。

        Returns:
            分类集合。
        """
        return {m.category for m in self._factors.values()}

    def tags(self) -> Set[str]:
        """获取所有标签。

        Returns:
            标签集合。
        """
        result: Set[str] = set()
        for m in self._factors.values():
            result.update(m.tags)
        return result

    def count(self) -> int:
        """获取因子总数。"""
        return len(self._factors)

    # ------------------------------------------------------------------
    # 计算
    # ------------------------------------------------------------------

    def compute(
        self, name: str, context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """执行因子计算。

        Args:
            name: 因子名。
            context: 计算上下文。

        Returns:
            计算结果。

        Raises:
            KeyError: 因子不存在或无计算函数。
        """
        key = name.upper()
        if key not in self._factors:
            raise KeyError(f"因子不存在: {name}")
        if key not in self._computors:
            raise KeyError(f"因子无计算函数: {name}")
        return self._computors[key](context or {})

    def has_computor(self, name: str) -> bool:
        """检查因子是否绑定了计算函数。"""
        return name.upper() in self._computors

    def __len__(self) -> int:
        return len(self._factors)

    def __contains__(self, name: str) -> bool:
        return self.exists(name)

    def __repr__(self) -> str:
        return f"FactorRegistry(count={len(self._factors)}, categories={len(self.categories())})"
