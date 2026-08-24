"""RAG 核心引擎：解析 → 语义分块 → 向量化 → 索引 → 多路融合检索。"""

from pathlib import Path
from typing import Any

from src.core.chunker import SemanticChunker
from src.core.config import Settings, get_settings
from src.core.embedding import EmbeddingGenerator
from src.core.fusion import reciprocal_rank_fusion
from src.core.indexer import VectorIndexer
from src.core.knowledge_graph import RuleKnowledgeGraph
from src.core.parser import PDFParser
from src.core.protocols import (
    ChunkerProtocol,
    EmbedderProtocol,
    KnowledgeGraphProtocol,
    PDFParserProtocol,
    QueryRewriterProtocol,
    VectorIndexerProtocol,
)
from src.core.query_rewriter import DeepSeekQueryRewriter
from src.utils.logger import logger


class RAGEngine:
    """协调解析、分块、嵌入、索引与多路融合检索；组件全部构造注入。"""

    def __init__(
        self,
        collection_name: str = "default",
        *,
        parser: PDFParserProtocol | None = None,
        embedder: EmbedderProtocol | None = None,
        indexer: VectorIndexerProtocol | None = None,
        chunker: ChunkerProtocol | None = None,
        query_rewriter: QueryRewriterProtocol | None = None,
        knowledge_graph: KnowledgeGraphProtocol | None = None,
        config: Settings | None = None,
    ) -> None:
        cfg = config or get_settings()
        self.collection_name = collection_name

        self.parser = parser or PDFParser(config=cfg)
        self.embedder = embedder or EmbeddingGenerator(config=cfg)
        self.indexer = indexer or VectorIndexer(
            collection_name=collection_name, config=cfg
        )
        self.chunker = chunker or SemanticChunker(
            max_tokens=cfg.chunk_max_tokens,
            overlap_ratio=cfg.chunk_overlap_ratio,
        )
        self.query_rewriter = query_rewriter or DeepSeekQueryRewriter(
            config=cfg, embedder=self.embedder
        )
        self.knowledge_graph = knowledge_graph or RuleKnowledgeGraph()

        self._rewrite_enabled = cfg.query_rewrite_enabled
        self._kg_enabled = cfg.kg_enabled
        self._kg_path = (
            Path(cfg.chroma_persist_dir) / f"{collection_name}.kg.json"
        )
        if self._kg_enabled:
            self.knowledge_graph.load(self._kg_path)
        self._is_indexed = False

    # ---------- 索引 ----------

    def build_index(self, pdf_path: str) -> dict[str, Any]:
        """解析 PDF → 语义分块 → 向量化 → 写索引，并构建知识图谱。"""
        logger.info(f"构建索引: {pdf_path}")

        pages = self.parser.parse(pdf_path)
        chunks = self.chunker.chunk(pages)
        if not chunks:
            raise ValueError("PDF中未提取到任何内容块")

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        self.indexer.clear()
        self.indexer.add(chunks, embeddings)
        self._is_indexed = True

        if self._kg_enabled:
            self.knowledge_graph.build(chunks)
            self.knowledge_graph.save(self._kg_path)

        logger.info(f"索引完成: {len(chunks)} 个分块, {len(pages)} 页")
        return {
            "total_chunks": len(chunks),
            "pages": len(pages),
            "collection": self.collection_name,
        }

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """多路检索：改写查询（或回退原始查询）向量检索 + 图谱候选，RRF 融合。"""
        if not self._is_indexed and self.indexer.count() == 0:
            raise RuntimeError("索引为空，请先构建索引")

        ranked_lists: list[list[dict[str, Any]]] = []

        # 1. 查询改写：有通过过滤的改写查询时不再使用原始查询
        rewritten = (
            self.query_rewriter.rewrite(query) if self._rewrite_enabled else []
        )
        search_queries = rewritten or [query]
        for search_query in search_queries:
            vector = self.embedder.embed([search_query])[0]
            ranked_lists.append(self.indexer.query(vector, top_k))

        # 2. 知识图谱候选
        if self._kg_enabled:
            kg_candidates = self.knowledge_graph.query_candidates(query, top_k)
            if kg_candidates:
                ranked_lists.append(kg_candidates)

        if not ranked_lists:
            return []
        return reciprocal_rank_fusion(ranked_lists, k=60, top_k=top_k)

    # ---------- 状态 ----------

    def get_stats(self) -> dict[str, Any]:
        return {
            "collection": self.collection_name,
            "total_vectors": self.indexer.count(),
            "is_indexed": self._is_indexed,
        }

    def clear(self) -> None:
        """清空索引（保留知识图谱文件，重建时会覆盖）。"""
        self.indexer.clear()
        self._is_indexed = False

    def drop_index(self) -> None:
        """删除索引与知识图谱文件。"""
        self.indexer.drop()
        self._kg_path.unlink(missing_ok=True)
        self._is_indexed = False
