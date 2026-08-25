"""RAG 核心引擎：解析 → 语义分块 → 向量化 → 索引 → 多路融合检索。"""

from pathlib import Path
from typing import Any

from src.core.bm25_retriever import BM25Retriever
from src.core.chunker import SemanticChunker
from src.core.config import Settings, get_settings
from src.core.embedding import EmbeddingGenerator
from src.core.fusion import item_key, reciprocal_rank_fusion
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
from src.core.reranker import LLMReranker
from src.utils.helpers import compute_confidence
from src.utils.logger import logger
from src.utils.math_normalize import normalize_math_text
from src.utils.query_routing import is_chapter_query, is_concept_query


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
        bm25: BM25Retriever | None = None,
        reranker: LLMReranker | None = None,
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
        self._candidate_k = cfg.retrieval_candidate_k
        self._bm25_enabled = cfg.bm25_enabled
        self._chapter_routing = cfg.chapter_query_routing
        self._confidence_threshold = cfg.answer_confidence_threshold
        self._llm_rerank_enabled = cfg.llm_rerank_enabled
        self._llm_rerank_top_n = cfg.llm_rerank_top_n
        self._subchunk_enabled = cfg.subchunk_enabled
        self._subchunk_max_tokens = cfg.subchunk_max_tokens

        self._bm25 = bm25
        self._reranker = reranker or (
            LLMReranker(config=cfg) if cfg.llm_rerank_enabled else None
        )
        self._kg_path = (
            Path(cfg.chroma_persist_dir) / f"{collection_name}.kg.json"
        )
        if self._kg_enabled:
            self.knowledge_graph.load(self._kg_path)
        self._is_indexed = False

    # ---------- 索引 ----------

    def build_index(self, pdf_path: str) -> dict[str, Any]:
        """解析 PDF → 分块 → （子块化）→ 向量化 → 写索引，并构建知识图谱。"""
        logger.info(f"构建索引: {pdf_path}")

        pages = self.parser.parse(pdf_path)
        chunks = self.chunker.chunk(pages)
        if not chunks:
            raise ValueError("PDF中未提取到任何内容块")

        if self._subchunk_enabled:
            subchunks = self.chunker.build_subchunks(
                chunks, self._subchunk_max_tokens
            )
            for sub in subchunks:
                sub["text"] = normalize_math_text(sub["text"])
            texts = [sub["text"] for sub in subchunks]
            embeddings = self.embedder.embed(texts)

            self.indexer.clear()
            self.indexer.add(subchunks, embeddings)
            self._is_indexed = True
            self._bm25 = None  # 语料已变化，下次检索时重建

            if self._kg_enabled:
                self.knowledge_graph.build(chunks)
                self.knowledge_graph.save(self._kg_path)

            logger.info(
                f"索引完成: {len(subchunks)} 个子块, "
                f"{len(chunks)} 个父块, {len(pages)} 页"
            )
            return {
                "total_chunks": len(subchunks),
                "parent_chunks": len(chunks),
                "pages": len(pages),
                "collection": self.collection_name,
            }

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        self.indexer.clear()
        self.indexer.add(chunks, embeddings)
        self._is_indexed = True
        self._bm25 = None

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
        """多路检索并返回融合结果（兼容旧接口）。"""
        return self._search(query, top_k)["results"]

    def search_with_confidence(
        self, query: str, top_k: int = 5
    ) -> dict[str, Any]:
        """多路检索 + 置信度与拒绝标记；拒绝时仍保留结果列表。"""
        outcome = self._search(query, top_k)
        threshold = self._confidence_threshold
        return {
            "results": outcome["results"],
            "confidence": round(outcome["confidence"], 4),
            "refused": threshold > 0 and outcome["confidence"] < threshold,
        }

    def _search(
        self, query: str, top_k: int
    ) -> dict[str, Any]:
        """多路检索：向量（改写变体/章节原始查询）+ 图谱 + BM25，RRF 融合。"""
        if not self._is_indexed and self.indexer.count() == 0:
            raise RuntimeError("索引为空，请先构建索引")

        norm_query = normalize_math_text(query)
        chapter_query = self._chapter_routing and is_chapter_query(norm_query)

        rewritten = (
            self.query_rewriter.rewrite(norm_query)
            if self._rewrite_enabled
            else []
        )
        # 章节类查询：只保留第 1 个同义改写变体，并加入原始查询兜底
        if chapter_query and rewritten:
            rewritten = rewritten[:1]
        search_queries = list(rewritten)
        if chapter_query:
            search_queries.append(norm_query)
        if not search_queries:
            search_queries = [norm_query]

        ranked_lists: list[list[dict[str, Any]]] = []
        max_cosine = 0.0
        for search_query in search_queries:
            vector = self.embedder.embed([search_query])[0]
            results = self.indexer.query(vector, self._candidate_k)
            if results:
                max_cosine = max(
                    max_cosine,
                    max(float(r.get("score") or 0.0) for r in results),
                )
            ranked_lists.append(results)

        if self._kg_enabled:
            kg_candidates = self.knowledge_graph.query_candidates(
                norm_query, self._candidate_k
            )
            if kg_candidates:
                ranked_lists.append(kg_candidates)

        if self._bm25_enabled:
            bm25_candidates = self._ensure_bm25().query(
                norm_query, self._candidate_k
            )
            if bm25_candidates:
                ranked_lists.append(bm25_candidates)

        if not ranked_lists:
            return {
                "results": [],
                "confidence": 0.0,
                "max_cosine": 0.0,
                "agree_legs": 0,
                "leg_count": 0,
            }

        fused = reciprocal_rank_fusion(ranked_lists, k=60)
        if self._reranker is not None and is_concept_query(norm_query):
            try:
                ordered = self._reranker.rerank(
                    norm_query, fused[: self._llm_rerank_top_n], top_k
                )
                results = self._restore_parent_text(ordered)
            except Exception as e:
                logger.warning(f"LLM 重排异常，回退原顺序: {e}")
                results = self._restore_parent_text(fused[:top_k])
        else:
            results = self._restore_parent_text(fused[:top_k])

        agree_legs = 0
        if fused:
            key = item_key(fused[0])
            agree_legs = sum(
                1 for leg in ranked_lists if any(item_key(item) == key for item in leg)
            )

        confidence = compute_confidence(max_cosine, agree_legs, len(ranked_lists))
        return {
            "results": results,
            "confidence": confidence,
            "max_cosine": max_cosine,
            "agree_legs": agree_legs,
            "leg_count": len(ranked_lists),
        }

    @staticmethod
    def _restore_parent_text(
        items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """命中子块时返回完整父块文本，并保留 fragment 为命中片段原文。"""
        restored: list[dict[str, Any]] = []
        for item in items:
            item = dict(item)
            parent_text = item.get("parent_text")
            if parent_text:
                item["fragment"] = item.get("text")
                item["text"] = parent_text
            restored.append(item)
        return restored

    def _ensure_bm25(self) -> BM25Retriever:
        """从当前索引集合懒构建 BM25 语料（含父块文本元数据）。"""
        if self._bm25 is None:
            retriever = BM25Retriever()
            try:
                snapshot = self.indexer.collection.get(
                    include=["documents", "metadatas"]
                )
            except Exception as e:
                logger.warning(f"BM25 语料加载失败: {e}")
                snapshot = {"documents": [], "metadatas": []}
            chunks = []
            for doc, meta in zip(
                snapshot.get("documents", []),
                snapshot.get("metadatas", []),
            ):
                item = dict(meta or {})
                item["text"] = doc
                chunks.append(item)
            retriever.build(chunks)
            self._bm25 = retriever
        return self._bm25

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
        self._bm25 = None
        self._is_indexed = False

    def drop_index(self) -> None:
        """删除索引与知识图谱文件。"""
        self.indexer.drop()
        self._kg_path.unlink(missing_ok=True)
        self._bm25 = None
        self._is_indexed = False
