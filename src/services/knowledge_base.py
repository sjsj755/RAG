"""知识库服务：管理多文档的注册表（持久化）与索引生命周期。"""

import json
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import chromadb

from src.core.answer_generator import LLMAnswerGenerator
from src.core.chunker import SemanticChunker
from src.core.config import Settings, get_settings
from src.core.embedding import EmbeddingGenerator
from src.core.knowledge_graph import RuleKnowledgeGraph
from src.core.models import DocumentInfo, DocumentStatus, GroupInfo
from src.core.rag_engine import RAGEngine
from src.services.document_repository import DocumentRepository
from src.services.group_repository import GroupRepository
from src.utils.helpers import (
    ensure_dir,
    generate_document_id,
    generate_group_id,
)
from src.utils.logger import logger
from src.utils.math_normalize import normalize_math_text


class KnowledgeBaseService:
    """多文档知识库服务。

    - 文档元数据通过 DocumentRepository 持久化，服务重启后不丢失；
    - 每个文档对应一个独立 Chroma 集合（与 doc_id 同名）；
    - 删除文档时同步删除集合与注册表记录。
    """

    def __init__(
        self,
        config: Settings | None = None,
        registry_path: str | Path | None = None,
        engine_factory: Callable[[str], RAGEngine] | None = None,
        group_engine_factory: Callable[[str], RAGEngine] | None = None,
        answer_generator: LLMAnswerGenerator | None = None,
        embedder: EmbeddingGenerator | None = None,
        reconcile: bool = True,
    ) -> None:
        self._config = config or get_settings()
        self._answer_generator = answer_generator or LLMAnswerGenerator(
            config=self._config
        )
        self._embedder = embedder or EmbeddingGenerator(config=self._config)
        self._repository = DocumentRepository(
            registry_path or self._config.registry_file
        )
        self._engine_factory = engine_factory or (
            lambda doc_id: RAGEngine(
                collection_name=doc_id, config=self._config
            )
        )
        self._group_engine_factory = group_engine_factory or (
            lambda group_id: RAGEngine(
                collection_name=group_id, config=self._config
            )
        )
        self._engines: dict[str, RAGEngine] = {}
        self._group_engines: dict[str, RAGEngine] = {}
        self._group_repository = GroupRepository(self._config.groups_file)
        self._groups: dict[str, GroupInfo] = self._group_repository.load()
        self._lock = threading.RLock()
        ensure_dir(self._config.upload_dir)
        self._documents: dict[str, DocumentInfo] = self._repository.load()
        if reconcile:
            self._cleanup_orphan_collections()

    def _cleanup_orphan_collections(self) -> None:
        """删除注册表中已不存在的 doc_* 集合，避免数据无限膨胀。"""
        try:
            client = chromadb.PersistentClient(
                path=self._config.chroma_persist_dir,
                settings=chromadb.config.Settings(anonymized_telemetry=False),
            )
            for collection in client.list_collections():
                name = getattr(collection, "name", str(collection))
                # 保留文档集合（doc_id 同名 / doc_<doc_id>）与分组集合
                legacy_key = name[4:] if name.startswith("doc_") else name
                if (
                    name in self._documents
                    or legacy_key in self._documents
                    or name in self._groups
                    or legacy_key in self._groups
                ):
                    continue
                try:
                    client.delete_collection(name)
                    logger.info(f"清理孤儿集合: {name}")
                except Exception:
                    logger.warning(f"孤儿集合 {name} 清理失败")
        except Exception as e:
            logger.warning(f"孤儿集合清理跳过: {e}")

    # ---------- 文档管理 ----------

    def create_document(self, filename: str, file_size: int) -> str:
        """创建文档记录并返回 doc_id。"""
        doc_id = generate_document_id(filename)
        now = datetime.now(UTC)
        with self._lock:
            self._documents[doc_id] = DocumentInfo(
                id=doc_id,
                filename=filename,
                file_size=file_size,
                status=DocumentStatus.PENDING,
                total_chunks=0,
                created_at=now,
                updated_at=now,
            )
            self._repository.save(self._documents)
        return doc_id

    def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus,
        total_chunks: int = 0,
        error: str | None = None,
    ) -> None:
        """更新文档状态并持久化。"""
        with self._lock:
            doc = self._documents.get(doc_id)
            if not doc:
                return
            doc.status = status
            doc.total_chunks = total_chunks
            doc.error = error
            doc.updated_at = datetime.now(UTC)
            self._repository.save(self._documents)

    def update_document_file_size(self, doc_id: str, file_size: int) -> None:
        """上传完成后回填实际文件字节数。"""
        with self._lock:
            doc = self._documents.get(doc_id)
            if not doc:
                return
            doc.file_size = file_size
            doc.updated_at = datetime.now(UTC)
            self._repository.save(self._documents)

    def get_document_info(self, doc_id: str) -> DocumentInfo | None:
        return self._documents.get(doc_id)

    def list_documents(self) -> list[DocumentInfo]:
        return sorted(
            self._documents.values(),
            key=lambda doc: doc.created_at,
            reverse=True,
        )

    def get_engine(self, doc_id: str) -> RAGEngine | None:
        """获取（或创建）文档对应的 RAG 引擎。"""
        if doc_id not in self._documents:
            return None
        with self._lock:
            engine = self._engines.get(doc_id)
            if engine is None:
                engine = self._engine_factory(doc_id)
                self._engines[doc_id] = engine
            return engine

    # ---------- 分组管理 ----------

    def create_group(self, name: str) -> GroupInfo:
        """创建命名分组并持久化。"""
        group_id = generate_group_id()
        now = datetime.now(UTC)
        group = GroupInfo(
            id=group_id,
            name=name,
            doc_ids=[],
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._groups[group_id] = group
            self._group_repository.save(self._groups)
        return group

    def list_groups(self) -> list[GroupInfo]:
        return sorted(
            self._groups.values(),
            key=lambda group: group.created_at,
            reverse=True,
        )

    def get_group(self, group_id: str) -> GroupInfo | None:
        return self._groups.get(group_id)

    def get_group_engine(self, group_id: str) -> RAGEngine | None:
        """获取（或创建）分组对应的共享集合引擎。"""
        if group_id not in self._groups:
            return None
        with self._lock:
            engine = self._group_engines.get(group_id)
            if engine is None:
                engine = self._group_engine_factory(group_id)
                self._group_engines[group_id] = engine
            return engine

    def _doc_kg_path(self, doc_id: str) -> Path:
        return Path(self._config.chroma_persist_dir) / f"{doc_id}.kg.json"

    def _group_kg_path(self, group_id: str) -> Path:
        return Path(self._config.chroma_persist_dir) / f"{group_id}.kg.json"

    def _save_groups(self) -> None:
        self._group_repository.save(self._groups)

    @staticmethod
    def _touch_group(group: GroupInfo) -> None:
        group.updated_at = datetime.now(UTC)

    def _rebuild_group_kg(self, group: GroupInfo) -> None:
        """从成员文档 KG 合并重建分组 KG，并刷新引擎内图谱。"""
        kg = RuleKnowledgeGraph()
        kg.load_many(
            [(doc_id, self._doc_kg_path(doc_id)) for doc_id in group.doc_ids],
            doc_names={
                doc_id: self._documents[doc_id].filename
                for doc_id in group.doc_ids
                if doc_id in self._documents
            },
        )
        kg.save(self._group_kg_path(group.id))
        engine = self._group_engines.get(group.id)
        if engine is not None:
            engine.knowledge_graph = kg

    def _drop_doc_collection(self, doc_id: str) -> None:
        """只删除文档的旧单文档集合（保留 KG 文件）。"""
        try:
            client = chromadb.PersistentClient(
                path=self._config.chroma_persist_dir,
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False
                ),
            )
            client.delete_collection(doc_id)
            logger.info(f"单文档集合 {doc_id} 已删除")
        except Exception:
            logger.warning(f"单文档集合 {doc_id} 删除失败或不存在")

    def _drop_group_collection(self, group_id: str) -> None:
        """删除分组集合与分组 KG 文件。"""
        try:
            client = chromadb.PersistentClient(
                path=self._config.chroma_persist_dir,
                settings=chromadb.config.Settings(
                    anonymized_telemetry=False
                ),
            )
            client.delete_collection(group_id)
            logger.info(f"分组集合 {group_id} 已删除")
        except Exception:
            logger.warning(f"分组集合 {group_id} 删除失败或不存在")
        self._group_kg_path(group_id).unlink(missing_ok=True)

    # ---------- 索引与检索 ----------

    def build_index(self, doc_id: str, pdf_path: str) -> dict:
        """为文档构建索引，任何失败都会将文档标记为 failed。"""
        engine = self.get_engine(doc_id)
        if not engine:
            raise ValueError(f"文档不存在: {doc_id}")

        self.update_document_status(doc_id, DocumentStatus.PROCESSING)

        try:
            result = engine.build_index(pdf_path)
        except Exception as e:
            self.update_document_status(
                doc_id, DocumentStatus.FAILED, error=str(e)
            )
            raise

        self.update_document_status(
            doc_id,
            DocumentStatus.COMPLETED,
            total_chunks=result["total_chunks"],
        )
        return result

    def index_into_group(
        self, group_id: str, doc_id: str, pdf_path: str
    ) -> dict:
        """把文档增量写入分组共享集合，并清理旧单文档集合。"""
        group = self._groups.get(group_id)
        if not group:
            raise ValueError(f"分组不存在: {group_id}")
        doc = self._documents.get(doc_id)
        if not doc:
            raise ValueError(f"文档不存在: {doc_id}")
        if doc_id in group.doc_ids:
            raise ValueError("文档已在分组中")
        engine = self.get_group_engine(group_id)
        if engine is None:
            raise ValueError(f"分组不存在: {group_id}")

        self.update_document_status(doc_id, DocumentStatus.PROCESSING)
        try:
            result = engine.add_document(pdf_path, doc_id, doc.filename)
        except Exception as e:
            self.update_document_status(
                doc_id, DocumentStatus.FAILED, error=str(e)
            )
            raise

        # 保存单文档 KG（父块），供合并图谱与后续迁移使用
        doc_kg = RuleKnowledgeGraph()
        for chunk in result["chunks"]:
            chunk["doc_id"] = doc_id
            chunk["filename"] = doc.filename
        doc_kg.build(result["chunks"])
        doc_kg.save(self._doc_kg_path(doc_id))

        with self._lock:
            if doc_id not in group.doc_ids:
                group.doc_ids.append(doc_id)
            self._touch_group(group)
            self._save_groups()
        self._rebuild_group_kg(group)
        engine.invalidate_bm25()

        # 清理旧单文档集合与引擎缓存（保留 KG）
        self._drop_doc_collection(doc_id)
        self._engines.pop(doc_id, None)

        self.update_document_status(
            doc_id,
            DocumentStatus.COMPLETED,
            total_chunks=result["total_chunks"],
        )
        logger.info(f"文档 {doc_id} 已编入分组 {group_id}")
        return {
            "group_id": group_id,
            "total_chunks": result["total_chunks"],
        }

    def migrate_group(self, group_id: str) -> dict:
        """把全部 completed 且未入组的文档基于 KG 数据迁移进分组。"""
        group = self._groups.get(group_id)
        if not group:
            raise ValueError(f"分组不存在: {group_id}")
        engine = self.get_group_engine(group_id)
        if engine is None:
            raise ValueError(f"分组不存在: {group_id}")

        candidates = [
            doc
            for doc in self._documents.values()
            if doc.status == DocumentStatus.COMPLETED
            and doc.id not in group.doc_ids
        ]
        chunker = SemanticChunker(
            max_tokens=self._config.chunk_max_tokens,
            overlap_ratio=self._config.chunk_overlap_ratio,
        )
        results: list[dict] = []
        for doc in candidates:
            try:
                kg_path = self._doc_kg_path(doc.id)
                if not kg_path.exists():
                    raise ValueError("缺少 KG 数据，请先正常索引")
                data = json.loads(kg_path.read_text(encoding="utf-8"))
                parent_chunks = list(data.get("chunks", {}).values())
                if not parent_chunks:
                    raise ValueError("KG 中无内容块")
                for parent in parent_chunks:
                    parent["doc_id"] = doc.id
                    parent["filename"] = doc.filename
                subchunks = chunker.build_subchunks(
                    parent_chunks, self._config.subchunk_max_tokens
                )
                for sub in subchunks:
                    sub["text"] = normalize_math_text(sub["text"])
                    sub["doc_id"] = doc.id
                    sub["filename"] = doc.filename
                texts = [sub["text"] for sub in subchunks]
                embeddings = self._embedder.embed(texts)
                engine.indexer.add(subchunks, embeddings)
                # 回写带元数据的单文档 KG，保证后续合并图谱完整
                fresh_kg = RuleKnowledgeGraph()
                fresh_kg.build(parent_chunks)
                fresh_kg.save(kg_path)
                results.append(
                    {
                        "doc_id": doc.id,
                        "status": "migrated",
                        "total_chunks": len(subchunks),
                    }
                )
            except Exception as e:
                logger.exception(f"迁移文档失败 {doc.id}: {e}")
                results.append(
                    {"doc_id": doc.id, "status": "failed", "error": str(e)}
                )

        migrated = [r for r in results if r["status"] == "migrated"]
        with self._lock:
            for result in migrated:
                if result["doc_id"] not in group.doc_ids:
                    group.doc_ids.append(result["doc_id"])
            self._touch_group(group)
            self._save_groups()
        self._rebuild_group_kg(group)
        engine.invalidate_bm25()
        for result in migrated:
            self._drop_doc_collection(result["doc_id"])
            self._engines.pop(result["doc_id"], None)

        return {
            "group_id": group_id,
            "total": len(candidates),
            "succeeded": len(migrated),
            "failed": len(candidates) - len(migrated),
            "results": results,
        }

    def group_remove_document(self, group_id: str, doc_id: str) -> None:
        """把文档移出分组：删除组内向量并标记文档为 pending。"""
        group = self._groups.get(group_id)
        if not group:
            raise ValueError(f"分组不存在: {group_id}")
        if doc_id not in group.doc_ids:
            raise ValueError(f"文档不在分组中: {doc_id}")
        engine = self.get_group_engine(group_id)
        if engine is None:
            raise ValueError(f"分组不存在: {group_id}")

        engine.indexer.delete_by_doc_id(doc_id)
        engine.invalidate_bm25()
        with self._lock:
            group.doc_ids.remove(doc_id)
            self._touch_group(group)
            self._save_groups()
        self._rebuild_group_kg(group)
        self.update_document_status(doc_id, DocumentStatus.PENDING)

    def delete_group(self, group_id: str) -> None:
        """删除分组：移除组集合/KG/注册表，并把成员文档标记为 pending。"""
        group = self._groups.get(group_id)
        if not group:
            raise ValueError(f"分组不存在: {group_id}")
        with self._lock:
            self._group_engines.pop(group_id, None)
            for doc_id in group.doc_ids:
                self.update_document_status(doc_id, DocumentStatus.PENDING)
            self._drop_group_collection(group_id)
            self._groups.pop(group_id, None)
            self._save_groups()

    def search(self, doc_id: str, query: str, top_k: int = 5) -> list[dict]:
        """在文档中检索。"""
        engine = self.get_engine(doc_id)
        if not engine:
            raise ValueError(f"文档不存在: {doc_id}")
        return engine.search(query, top_k)

    def search_with_confidence(
        self, doc_id: str, query: str, top_k: int = 5
    ) -> dict:
        """在文档中检索并返回置信度与拒绝标记。"""
        engine = self.get_engine(doc_id)
        if not engine:
            raise ValueError(f"文档不存在: {doc_id}")
        return engine.search_with_confidence(query, top_k)

    def answer(self, doc_id: str, query: str, top_k: int = 5) -> dict:
        """检索 + LLM 生成答案；低置信度时短路拒绝，不调用 LLM。"""
        return self._answer_outcome(
            self.search_with_confidence(doc_id, query, top_k), query
        )

    def _answer_outcome(self, outcome: dict, query: str) -> dict:
        """基于检索结果生成答案（供单文档与分组复用）。"""
        if outcome["refused"]:
            return {
                "query": query,
                "answer": None,
                "refused": True,
                "refusal_reason": "检索置信度不足，未生成答案",
                "confidence": outcome["confidence"],
                "sources": [],
            }
        sources = self._build_sources(outcome["results"])
        answer = self._answer_generator.generate(query, sources)
        return {
            "query": query,
            "answer": answer,
            "refused": False,
            "refusal_reason": None,
            "confidence": outcome["confidence"],
            "sources": sources,
        }

    def stream_answer(
        self, doc_id: str, query: str, top_k: int = 5
    ) -> Iterator[dict]:
        """流式答案事件序列：sources → answer×n → done；低置信为 refused + done。"""
        yield from self._stream_outcome(
            self.search_with_confidence(doc_id, query, top_k), query
        )

    def _stream_outcome(
        self, outcome: dict, query: str
    ) -> Iterator[dict]:
        """基于检索结果产出流式答案事件（供单文档与分组复用）。"""
        if outcome["refused"]:
            yield {
                "type": "refused",
                "reason": "检索置信度不足，未生成答案",
                "confidence": outcome["confidence"],
            }
            yield {"type": "done", "refused": True, "confidence": outcome["confidence"]}
            return

        sources = self._build_sources(outcome["results"])
        yield {
            "type": "sources",
            "sources": sources,
            "confidence": outcome["confidence"],
        }
        for piece in self._answer_generator.stream(query, sources):
            yield {"type": "answer", "content": piece}
        yield {"type": "done", "refused": False, "confidence": outcome["confidence"]}

    # ---------- 分组检索与问答 ----------

    def group_search_with_confidence(
        self, group_id: str, query: str, top_k: int = 5
    ) -> dict:
        """在分组共享集合中检索并返回置信度。"""
        engine = self.get_group_engine(group_id)
        if not engine:
            raise ValueError(f"分组不存在: {group_id}")
        return engine.search_with_confidence(query, top_k)

    def group_answer(
        self, group_id: str, query: str, top_k: int = 5
    ) -> dict:
        """分组检索 + LLM 生成答案。"""
        return self._answer_outcome(
            self.group_search_with_confidence(group_id, query, top_k), query
        )

    def group_stream_answer(
        self, group_id: str, query: str, top_k: int = 5
    ) -> Iterator[dict]:
        """分组流式答案事件序列。"""
        yield from self._stream_outcome(
            self.group_search_with_confidence(group_id, query, top_k), query
        )

    @staticmethod
    def _build_sources(results: list[dict]) -> list[dict]:
        """把检索结果组装为引用来源（fragment 优先，回退父块文本）。"""
        sources = []
        for i, result in enumerate(results, start=1):
            text = result.get("fragment") or result.get("text") or ""
            sources.append(
                {
                    "index": i,
                    "page_num": result.get("page_num", 0),
                    "text": text,
                    "score": result.get("score"),
                    "doc_id": result.get("doc_id"),
                    "filename": result.get("filename"),
                }
            )
        return sources

    def delete_document(self, doc_id: str) -> None:
        """删除文档：移除索引集合、引擎实例与注册表记录。"""
        with self._lock:
            engine = self._engines.pop(doc_id, None)
            if engine is not None:
                engine.drop_index()
            else:
                # 引擎不在内存缓存（如服务重启后）时，直接删除集合与图谱文件
                self._drop_collection(doc_id)
            self._documents.pop(doc_id, None)
            self._repository.save(self._documents)

    def _drop_collection(self, doc_id: str) -> None:
        """直接删除文档对应的 Chroma 集合与知识图谱文件。"""
        try:
            client = chromadb.PersistentClient(
                path=self._config.chroma_persist_dir,
                settings=chromadb.config.Settings(anonymized_telemetry=False),
            )
            client.delete_collection(doc_id)
            logger.info(f"集合 {doc_id} 已删除")
        except Exception:
            logger.warning(f"集合 {doc_id} 删除失败或不存在")
        kg_path = (
            Path(self._config.chroma_persist_dir) / f"{doc_id}.kg.json"
        )
        kg_path.unlink(missing_ok=True)
