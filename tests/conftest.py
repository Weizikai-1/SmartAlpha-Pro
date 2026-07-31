"""共享测试 fixtures。

为所有测试模块提供公共数据和对象的 pytest fixtures。
"""

import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from smartalpha.core.lexer import ExpressionLexer, Token, TokenType
from smartalpha.core.parser import ExpressionParser
from smartalpha.core.executor import ASTExecutor
from smartalpha.core.functions import FinancialFunctionLibrary
from smartalpha.storage.cache import LRUCache, CacheStats
from smartalpha.storage.columnar import BinStorage
from smartalpha.registry.factor import FactorRegistry, FactorMetadata
from smartalpha.registry.dependency import FactorDependencyGraph


@pytest.fixture
def rng():
    """固定随机数生成器，确保测试可复现。"""
    return np.random.RandomState(42)


@pytest.fixture
def sample_1d(rng):
    """生成一维测试数据 (200 个交易日)。"""
    n = 200
    return {
        "close": rng.randn(n).cumsum() + 100,
        "open": rng.randn(n).cumsum() + 100,
        "high": rng.randn(n).cumsum() + 102,
        "low": rng.randn(n).cumsum() + 98,
        "volume": rng.randint(1_000_000, 10_000_000, n).astype(float),
    }


@pytest.fixture
def small_data():
    """小型测试数据 (10 个数据点)。"""
    return {
        "close": np.array([10.0, 11.0, 12.0, 11.5, 13.0, 12.5, 14.0, 13.5, 15.0, 14.5]),
        "open": np.array([9.5, 10.5, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0]),
        "volume": np.array([1000.0, 2000.0, 1500.0, 3000.0, 2500.0, 1800.0, 2200.0, 3100.0, 2800.0, 1600.0]),
    }


@pytest.fixture
def lexer():
    """创建词法分析器实例。"""
    return ExpressionLexer()


@pytest.fixture
def parser():
    """创建语法分析器实例。"""
    return ExpressionParser()


@pytest.fixture
def executor():
    """创建 AST 执行器实例。"""
    return ASTExecutor()


@pytest.fixture
def func_lib():
    """创建金融函数库实例。"""
    return FinancialFunctionLibrary()


@pytest.fixture
def cache():
    """创建默认 LRU 缓存实例。"""
    return LRUCache(max_size=256)


@pytest.fixture
def storage_dir():
    """创建临时存储目录，测试结束后自动清理。"""
    d = tempfile.mkdtemp()
    yield d
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def registry():
    """创建因子注册表实例。"""
    return FactorRegistry()


@pytest.fixture
def dep_graph():
    """创建因子依赖图实例。"""
    return FactorDependencyGraph()