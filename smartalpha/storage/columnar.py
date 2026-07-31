"""列式存储模块。

提供基于 pickle 的列式（按列存取）持久化存储，
适用于因子数据的高效读写场景。

设计思路：
- 每列数据独立序列化为 pickle 文件
- 通过列名索引快速定位文件
- 支持批量读写与增量更新
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class StorageInfo:
    """存储元信息。

    Attributes:
        path: 存储目录路径。
        columns: 已存储的列名列表。
        row_count: 行数（若所有列行数一致）。
    """

    path: str
    columns: List[str] = field(default_factory=list)
    row_count: int = 0


class BinStorage:
    """列式二进制存储。

    每列数据以独立 pickle 文件存储于指定目录下，
    支持高效的单列读写与批量操作。

    使用示例::

        storage = BinStorage("./factor_data")
        storage.write_column("close", np.array([10.0, 11.0, 12.0]))
        storage.write_column("volume", np.array([1000, 2000, 1500]))
        close = storage.read_column("close")
        info = storage.info()
    """

    _METADATA_FILENAME: str = "_metadata.pkl"

    def __init__(self, path: str) -> None:
        """初始化列式存储。

        Args:
            path: 存储目录路径，不存在时自动创建。
        """
        self._path: Path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._metadata: Dict[str, Any] = self._load_metadata()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    @property
    def path(self) -> str:
        return str(self._path)

    def list_columns(self) -> List[str]:
        """列出所有列名。

        Returns:
            列名列表。
        """
        columns = set(self._metadata.get("columns", []))
        # 同时扫描磁盘上的实际文件
        for f in self._path.iterdir():
            if f.suffix == ".pkl" and f.stem != self._METADATA_FILENAME.replace(".pkl", ""):
                columns.add(f.stem.replace("_", "/"))
        return sorted(columns)

    def read_column(self, name: str) -> np.ndarray:
        """读取指定列的数据。

        Args:
            name: 列名。

        Returns:
            numpy 数组。

        Raises:
            FileNotFoundError: 列不存在时。
        """
        filepath = self._column_path(name)
        if not filepath.exists():
            raise FileNotFoundError(f"列 '{name}' 不存在于 {self._path}")

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, np.ndarray):
            return data
        return np.asarray(data)

    def write_column(self, name: str, data: Any) -> None:
        """写入一列数据。

        Args:
            name: 列名。
            data: 数据（数组、列表等可序列化对象）。
        """
        arr = np.asarray(data, dtype=np.float64)
        filepath = self._column_path(name)

        with open(filepath, "wb") as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)

        self._metadata.setdefault("columns", [])
        if name not in self._metadata["columns"]:
            self._metadata["columns"].append(name)
        self._save_metadata()

    def write_batch(
        self, columns: Dict[str, Any]
    ) -> None:
        """批量写入多列。

        Args:
            columns: 列名 → 数据 的映射。
        """
        for name, data in columns.items():
            self.write_column(name, data)

    def read_batch(
        self, names: Iterable[str]
    ) -> Dict[str, np.ndarray]:
        """批量读取多列。

        Args:
            names: 列名可迭代对象。

        Returns:
            列名 → 数组 的映射。
        """
        result: Dict[str, np.ndarray] = {}
        for name in names:
            try:
                result[name] = self.read_column(name)
            except FileNotFoundError:
                pass
        return result

    def delete_column(self, name: str) -> bool:
        """删除指定列。

        Args:
            name: 列名。

        Returns:
            是否成功删除。
        """
        filepath = self._column_path(name)
        if filepath.exists():
            filepath.unlink()
            columns = self._metadata.get("columns", [])
            if name in columns:
                columns.remove(name)
                self._metadata["columns"] = columns
                self._save_metadata()
            return True
        return False

    def column_exists(self, name: str) -> bool:
        """检查列是否存在。

        Args:
            name: 列名。

        Returns:
            是否存在。
        """
        return self._column_path(name).exists()

    def info(self) -> StorageInfo:
        """获取存储元信息。

        Returns:
            StorageInfo 对象。
        """
        columns = self.list_columns()
        row_count = 0
        if columns:
            try:
                first_col = self.read_column(columns[0])
                row_count = len(first_col)
            except Exception:
                import logging
                logging.getLogger(__name__).warning(f"读取列 {columns[0]} 失败")
        return StorageInfo(
            path=str(self._path),
            columns=columns,
            row_count=row_count,
        )

    def clear(self) -> None:
        """清空所有列数据。"""
        for f in self._path.iterdir():
            if f.suffix == ".pkl":
                f.unlink()
        self._metadata = {}
        self._save_metadata()

    # ------------------------------------------------------------------
    # 持久化辅助
    # ------------------------------------------------------------------

    def _column_path(self, name: str) -> Path:
        """获取列文件路径。"""
        # 清理非法文件名字符
        safe_name = name.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self._path / f"{safe_name}.pkl"

    def _metadata_path(self) -> Path:
        """获取元信息文件路径。"""
        return self._path / self._METADATA_FILENAME

    def _load_metadata(self) -> Dict[str, Any]:
        """加载元信息。"""
        path = self._metadata_path()
        if path.exists():
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                import logging
                logging.getLogger(__name__).warning(f"元信息加载失败: {self._metadata_path()}")
                return {}
        return {}

    def _save_metadata(self) -> None:
        """保存元信息。"""
        with open(self._metadata_path(), "wb") as f:
            pickle.dump(self._metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
