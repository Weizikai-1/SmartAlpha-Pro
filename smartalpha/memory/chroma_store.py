"""ChromaDB 反思记忆存储 — 支撑 Agent RAG 增强。

提供 3 个 Collection:
- agent_decisions:  历史决策 (信号+置信度+推理)
- market_patterns:  市场快照特征向量 (用于相似场景检索)
- reflections:      反思记录 (成功/失败原因)

所有写入/查询均带容错降级，ChromaDB 不可用时不影响主流程。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ChromaDB 默认持久化目录
_DEFAULT_PERSIST_DIR = os.path.join(
    os.path.expanduser("~"), ".smartalpha", "chroma"
)


class ChromaMemoryStore:
    """ChromaDB 向量存储封装 — Agent 反思记忆。

    设计原则:
    - 可插拔: ChromaDB 不可用时自动降级，不抛异常
    - 独立: 不修改现有任何模块
    - 轻量: 默认本地持久化，无需外部服务

    使用:
        store = ChromaMemoryStore()
        store.store_decision("000001.SZ", "fundamental",
                             {"signal": "bullish", "confidence": 0.8},
                             {"date": "20240701"})
        similar = store.query_similar("market_patterns", "000001.SZ 银行", k=3)
    """

    def __init__(self, persist_dir: Optional[str] = None):
        self._persist_dir = persist_dir or _DEFAULT_PERSIST_DIR
        self._client = None
        self._collections: dict[str, Any] = {}
        self._available = False

        try:
            self._init_client()
        except Exception as e:
            logger.warning(f"ChromaDB 初始化失败，记忆功能降级: {e}")

    def _init_client(self) -> None:
        """初始化 ChromaDB 客户端并确保 Collection 存在。"""
        try:
            import chromadb
            os.makedirs(self._persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)

            # 创建/加载 3 个 Collection
            for name in ["agent_decisions", "market_patterns", "reflections"]:
                try:
                    self._collections[name] = self._client.get_collection(name)
                except Exception:
                    self._collections[name] = self._client.create_collection(
                        name=name,
                        metadata={"description": f"SmartAlpha {name} 向量存储"},
                    )
            self._available = True
            logger.info(f"ChromaDB 就绪: {self._persist_dir}")
        except ImportError:
            logger.warning("chromadb 未安装 — 记忆功能不可用，安装: pip install chromadb")
        except Exception as e:
            logger.warning(f"ChromaDB 连接失败: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # 决策存储
    # ------------------------------------------------------------------

    def store_decision(
        self,
        stock_code: str,
        agent_name: str,
        output: dict,
        metadata: Optional[dict] = None,
    ) -> bool:
        """存储一次 Agent 决策。

        Args:
            stock_code: Tushare 格式代码
            agent_name: Agent 名称 (fundamental/technical/sentiment/news)
            output: AgentOutput.to_dict() 字典
            metadata: 额外元数据 (date, depth 等)
        """
        if not self._available:
            return False
        try:
            doc_id = f"{stock_code}_{agent_name}_{datetime.now().isoformat()}"
            text = (
                f"股票:{stock_code} Agent:{agent_name} "
                f"信号:{output.get('signal','?')} 置信度:{output.get('confidence',0)} "
                f"分析:{output.get('reasoning','')[:200]}"
            )
            meta = dict(metadata or {})
            meta.update({
                "stock_code": stock_code,
                "agent_name": agent_name,
                "timestamp": datetime.now().isoformat(),
            })
            self._collections["agent_decisions"].add(
                ids=[doc_id],
                documents=[text],
                metadatas=[meta],
            )
            return True
        except Exception as e:
            logger.debug(f"ChromaDB 决策存储失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 市场快照
    # ------------------------------------------------------------------

    def store_market_snapshot(
        self,
        stock_code: str,
        features: np.ndarray,
        metadata: Optional[dict] = None,
    ) -> bool:
        """存储市场快照的特征向量。

        Args:
            stock_code: 股票代码
            features: 特征向量 (float32 数组)
            metadata: 元数据
        """
        if not self._available:
            return False
        try:
            doc_id = f"{stock_code}_{datetime.now().isoformat()}"
            feat_list = features.astype(np.float32).tolist()
            if isinstance(feat_list, float):
                feat_list = [feat_list]

            meta = dict(metadata or {})
            meta.update({
                "stock_code": stock_code,
                "timestamp": datetime.now().isoformat(),
            })

            self._collections["market_patterns"].add(
                ids=[doc_id],
                embeddings=[feat_list],
                documents=[f"Market snapshot for {stock_code}"],
                metadatas=[meta],
            )
            return True
        except Exception as e:
            logger.debug(f"ChromaDB 快照存储失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 反思记录
    # ------------------------------------------------------------------

    def store_reflection(
        self,
        stock_code: str,
        prediction: str,
        actual_outcome: str,
        lesson: str,
    ) -> bool:
        """存储一条反思记录。

        Args:
            stock_code: 股票代码
            prediction: 当时预测方向
            actual_outcome: 实际结果
            lesson: 经验教训
        """
        if not self._available:
            return False
        try:
            doc_id = f"reflection_{stock_code}_{datetime.now().isoformat()}"
            text = f"预测:{prediction} → 实际:{actual_outcome} | 教训:{lesson}"
            self._collections["reflections"].add(
                ids=[doc_id],
                documents=[text],
                metadatas=[{
                    "stock_code": stock_code,
                    "prediction": prediction,
                    "actual": actual_outcome,
                    "timestamp": datetime.now().isoformat(),
                }],
            )
            return True
        except Exception as e:
            logger.debug(f"ChromaDB 反思存储失败: {e}")
            return False

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def query_similar(
        self,
        collection: str,
        query_text: str,
        k: int = 5,
    ) -> str:
        """检索相似历史场景，返回格式化文本。

        Args:
            collection: Collection 名称
            query_text: 查询文本
            k: 返回条数

        Returns:
            格式化文本，可直接 append 到 LLM prompt。
        """
        if not self._available or collection not in self._collections:
            return ""

        try:
            results = self._collections[collection].query(
                query_texts=[query_text],
                n_results=min(k, 100),
            )
            if not results or not results.get("documents") or not results["documents"][0]:
                return ""

            docs = results["documents"][0]
            lines = [f"[{collection}] 相似历史记录 ({len(docs)} 条):"]
            for i, doc in enumerate(docs, 1):
                lines.append(f"  {i}. {doc[:200]}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"ChromaDB 查询失败: {e}")
            return ""

    def query_by_stock(
        self,
        collection: str,
        stock_code: str,
        k: int = 10,
    ) -> list[dict]:
        """按股票代码查询历史记录。

        Args:
            collection: Collection 名称
            stock_code: 股票代码
            k: 返回条数

        Returns:
            list of {document, metadata} dicts
        """
        if not self._available or collection not in self._collections:
            return []
        try:
            results = self._collections[collection].get(
                where={"stock_code": stock_code},
                limit=k,
            )
            if not results or not results.get("documents"):
                return []
            return [
                {"document": doc, "metadata": meta}
                for doc, meta in zip(results["documents"], results["metadatas"] or [])
            ]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def count(self, collection: str) -> int:
        """返回 Collection 中的记录数。"""
        if not self._available or collection not in self._collections:
            return 0
        try:
            return self._collections[collection].count()
        except Exception:
            return 0

    def stats(self) -> dict:
        """返回各 Collection 的统计信息。"""
        return {
            name: self.count(name)
            for name in ["agent_decisions", "market_patterns", "reflections"]
        }
