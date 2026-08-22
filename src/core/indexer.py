"""向量索引器：基于 ChromaDB 持久化存储文本块与向量。"""

from typing import Any

import chromadb

from src.core.config import settings
from src.utils.helpers import ensure_dir
from src.utils.logger import logger


class VectorIndexer:
    def __init__(self, collection_name: str = "default") -> None:
        self.collection_name = collection_name

        # 确保持久化目录存在
        ensure_dir(settings.chroma_persist_dir)

        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """获取已存在的集合，不存在则创建。"""
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        """清空集合（删除后重建）。"""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            # 集合不存在时无需处理
            pass
        self._ensure_collection()
        logger.info(f"集合 {self.collection_name} 已清空")

    def drop(self) -> None:
        """彻底删除集合（不重建）。"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"集合 {self.collection_name} 已删除")
        except Exception:
            # 集合不存在时无需处理
            pass

    def add(
        self, chunks: list[dict[str, Any]], embeddings: list[list[float]]
    ) -> None:
        """添加内容块与向量。"""
        if not chunks:
            logger.warning("没有内容块可添加")
            return
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"内容块数量({len(chunks)})与向量数量({len(embeddings)})不一致"
            )

        texts = [c["text"] for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {k: v for k, v in c.items() if k != "text"} for c in chunks
        ]

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(
            f"添加 {len(chunks)} 个向量到集合 {self.collection_name}"
        )

    def query(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        """向量检索，返回带 score 的内容块。"""
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                item = {
                    "text": results["documents"][0][i],
                    "score": 1 - results["distances"][0][i],
                }
                item.update(results["metadatas"][0][i])
                retrieved.append(item)
        return retrieved

    def count(self) -> int:
        return self.collection.count()
