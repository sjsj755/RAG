"""RAG 核心引擎：协调解析、嵌入、索引。"""

from typing import Any

from src.core.config import Settings, get_settings
from src.core.embedding import EmbeddingGenerator
from src.core.indexer import VectorIndexer
from src.core.parser import PDFParser
from src.core.protocols import (
    EmbedderProtocol,
    PDFParserProtocol,
    VectorIndexerProtocol,
)
from src.utils.logger import logger


class RAGEngine:
    """协调解析、嵌入、索引；组件通过构造注入，便于测试与替换实现。"""

    def __init__(
        self,
        collection_name: str = "default",
        *,
        parser: PDFParserProtocol | None = None,
        embedder: EmbedderProtocol | None = None,
        indexer: VectorIndexerProtocol | None = None,
        config: Settings | None = None,
    ) -> None:
        cfg = config or get_settings()
        self.collection_name = collection_name
        self.parser = parser or PDFParser(config=cfg)
        self.embedder = embedder or EmbeddingGenerator(config=cfg)
        self.indexer = indexer or VectorIndexer(
            collection_name=collection_name, config=cfg
        )
        self._is_indexed = False

    def build_index(self, pdf_path: str) -> dict[str, Any]:
        """从 PDF 构建索引。"""
        logger.info(f"构建索引: {pdf_path}")

        # 1. 解析
        pages = self.parser.parse(pdf_path)

        # 2. 收集所有块
        all_chunks: list[dict[str, Any]] = []
        for page in pages:
            all_chunks.extend(page["blocks"])

        if not all_chunks:
            raise ValueError("PDF中未提取到任何内容块")

        # 3. 生成向量
        texts = [c["text"] for c in all_chunks]
        embeddings = self.embedder.embed(texts)

        # 4. 清空旧索引并添加新数据
        self.indexer.clear()
        self.indexer.add(all_chunks, embeddings)
        self._is_indexed = True

        return {
            "total_chunks": len(all_chunks),
            "pages": len(pages),
            "collection": self.collection_name,
        }

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """语义检索。"""
        if not self._is_indexed and self.indexer.count() == 0:
            raise RuntimeError("索引为空，请先构建索引")

        query_vector = self.embedder.embed([query])[0]
        return self.indexer.query(query_vector, top_k)

    def get_stats(self) -> dict[str, Any]:
        """获取索引统计。"""
        return {
            "collection": self.collection_name,
            "total_vectors": self.indexer.count(),
            "is_indexed": self._is_indexed,
        }

    def clear(self) -> None:
        """清空索引。"""
        self.indexer.clear()
        self._is_indexed = False
