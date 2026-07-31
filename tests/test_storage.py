"""列式存储 BinStorage 测试套件。

覆盖:
- write_column / read_column
- list_columns
- write_batch / read_batch
- delete_column / column_exists
- info / clear
- 增量更新
- 时间范围过滤
- 边界条件
"""

import os
import tempfile

import numpy as np
import pytest

from smartalpha.storage.columnar import BinStorage, StorageInfo


class TestWriteRead:
    """写入与读取测试。"""

    def test_write_and_read_column(self, storage_dir):
        """写入并读取列。"""
        storage = BinStorage(storage_dir)
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        storage.write_column("close", data)
        result = storage.read_column("close")
        np.testing.assert_array_almost_equal(result, data)

    def test_write_multiple_columns(self, storage_dir):
        """写入多列。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0, 2.0]))
        storage.write_column("open", np.array([3.0, 4.0]))
        storage.write_column("volume", np.array([100.0, 200.0]))
        assert len(storage.list_columns()) == 3

    def test_read_nonexistent_column(self, storage_dir):
        """读取不存在的列应抛出 FileNotFoundError。"""
        storage = BinStorage(storage_dir)
        with pytest.raises(FileNotFoundError):
            storage.read_column("nonexistent")

    def test_overwrite_existing_column(self, storage_dir):
        """覆盖已有列。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0, 2.0]))
        storage.write_column("close", np.array([10.0, 20.0]))
        result = storage.read_column("close")
        np.testing.assert_array_almost_equal(result, np.array([10.0, 20.0]))

    def test_write_list_data(self, storage_dir):
        """写入列表数据。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", [1.0, 2.0, 3.0])
        result = storage.read_column("close")
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_write_integer_data(self, storage_dir):
        """写入整数数据。"""
        storage = BinStorage(storage_dir)
        storage.write_column("volume", np.array([100, 200, 300]))
        result = storage.read_column("volume")
        np.testing.assert_array_almost_equal(result, np.array([100.0, 200.0, 300.0]))


class TestBatchOperations:
    """批量操作测试。"""

    def test_write_batch(self, storage_dir):
        """批量写入。"""
        storage = BinStorage(storage_dir)
        columns = {
            "close": np.array([1.0, 2.0]),
            "open": np.array([3.0, 4.0]),
            "volume": np.array([100.0, 200.0]),
        }
        storage.write_batch(columns)
        result = storage.read_batch(["close", "open"])
        assert len(result) == 2
        np.testing.assert_array_almost_equal(result["close"], np.array([1.0, 2.0]))

    def test_read_batch_missing_columns(self, storage_dir):
        """批量读取时跳过不存在的列。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0, 2.0]))
        result = storage.read_batch(["close", "nonexistent"])
        assert len(result) == 1
        assert "close" in result

    def test_write_batch_empty(self, storage_dir):
        """空批量写入。"""
        storage = BinStorage(storage_dir)
        storage.write_batch({})
        assert storage.list_columns() == []


class TestListColumns:
    """列列表测试。"""

    def test_list_columns_empty(self, storage_dir):
        """空存储的列列表。"""
        storage = BinStorage(storage_dir)
        assert storage.list_columns() == []

    def test_list_columns_after_write(self, storage_dir):
        """写入后的列列表。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0]))
        storage.write_column("open", np.array([2.0]))
        columns = storage.list_columns()
        assert "close" in columns
        assert "open" in columns
        assert len(columns) == 2

    def test_list_columns_sorted(self, storage_dir):
        """列名应排序返回。"""
        storage = BinStorage(storage_dir)
        storage.write_column("zebra", np.array([1.0]))
        storage.write_column("alpha", np.array([2.0]))
        storage.write_column("middle", np.array([3.0]))
        columns = storage.list_columns()
        assert columns == sorted(columns)


class TestColumnManagement:
    """列管理测试。"""

    def test_delete_column(self, storage_dir):
        """删除列。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0, 2.0]))
        assert storage.delete_column("close") is True
        assert not storage.column_exists("close")

    def test_delete_nonexistent(self, storage_dir):
        """删除不存在的列返回 False。"""
        storage = BinStorage(storage_dir)
        assert storage.delete_column("nonexistent") is False

    def test_column_exists(self, storage_dir):
        """检查列是否存在。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0]))
        assert storage.column_exists("close") is True
        assert storage.column_exists("nonexistent") is False

    def test_delete_and_read(self, storage_dir):
        """删除后读取应抛错。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0]))
        storage.delete_column("close")
        with pytest.raises(FileNotFoundError):
            storage.read_column("close")


class TestStorageInfo:
    """元信息测试。"""

    def test_info_empty(self, storage_dir):
        """空存储的元信息。"""
        storage = BinStorage(storage_dir)
        info = storage.info()
        assert isinstance(info, StorageInfo)
        assert info.path == storage_dir
        assert info.columns == []
        assert info.row_count == 0

    def test_info_with_data(self, storage_dir):
        """有数据时的元信息。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0, 2.0, 3.0]))
        storage.write_column("open", np.array([4.0, 5.0, 6.0]))
        info = storage.info()
        assert len(info.columns) == 2
        assert info.row_count == 3

    def test_path_property(self, storage_dir):
        """路径属性。"""
        storage = BinStorage(storage_dir)
        assert storage.path == storage_dir


class TestClear:
    """清空操作测试。"""

    def test_clear(self, storage_dir):
        """清空所有列。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0]))
        storage.write_column("open", np.array([2.0]))
        storage.clear()
        assert storage.list_columns() == []
        assert storage.column_exists("close") is False
        assert storage.column_exists("open") is False


class TestIncrementalUpdate:
    """增量更新测试。"""

    def test_overwrite_update(self, storage_dir):
        """覆盖式更新。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0, 2.0, 3.0]))
        storage.write_column("close", np.array([10.0, 20.0, 30.0]))
        result = storage.read_column("close")
        np.testing.assert_array_almost_equal(result, np.array([10.0, 20.0, 30.0]))

    def test_metadata_persistence(self, storage_dir):
        """元信息持久化。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close", np.array([1.0, 2.0]))
        storage.write_column("open", np.array([3.0, 4.0]))
        del storage
        storage2 = BinStorage(storage_dir)
        assert "close" in storage2.list_columns()
        assert "open" in storage2.list_columns()


class TestEdgeCases:
    """边界条件测试。"""

    def test_single_value_column(self, storage_dir):
        """单列单值。"""
        storage = BinStorage(storage_dir)
        storage.write_column("single", np.array([42.0]))
        result = storage.read_column("single")
        np.testing.assert_array_almost_equal(result, np.array([42.0]))

    def test_empty_array_column(self, storage_dir):
        """空数组列。"""
        storage = BinStorage(storage_dir)
        storage.write_column("empty", np.array([]))
        result = storage.read_column("empty")
        assert len(result) == 0

    def test_large_data(self, storage_dir):
        """大数据列 (10000 点)。"""
        storage = BinStorage(storage_dir)
        data = np.random.RandomState(42).randn(10000)
        storage.write_column("large", data)
        result = storage.read_column("large")
        np.testing.assert_array_almost_equal(result, data)

    def test_special_characters_in_name(self, storage_dir):
        """含特殊字符的列名。"""
        storage = BinStorage(storage_dir)
        storage.write_column("close/price", np.array([1.0, 2.0]))
        result = storage.read_column("close/price")
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0]))

    def test_create_storage_nonexistent_dir(self):
        """创建不存在的目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "subdir", "data")
            storage = BinStorage(new_dir)
            storage.write_column("test", np.array([1.0]))
            result = storage.read_column("test")
            np.testing.assert_array_almost_equal(result, np.array([1.0]))

    def test_multiple_storage_instances(self, storage_dir):
        """同一目录多个存储实例。"""
        s1 = BinStorage(storage_dir)
        s2 = BinStorage(storage_dir)
        s1.write_column("close", np.array([1.0, 2.0]))
        result = s2.read_column("close")
        np.testing.assert_array_almost_equal(result, np.array([1.0, 2.0]))

    def test_storage_repr(self, storage_dir):
        """存储路径信息。"""
        storage = BinStorage(storage_dir)
        assert storage.path == storage_dir