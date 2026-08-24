"""知识库服务：管理多文档的注册表（持久化）与索引生命周期。"""

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import chromadb

from src.core.config import Settings, get_settings
from src.core.models import DocumentInfo, DocumentStatus
from src.core.rag_engine import RAGEngine
from src.services.document_repository import DocumentRepository
from src.utils.helpers import ensure_dir, generate_document_id
from src.utils.logger import logger


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
        reconcile: bool = True,
    ) -> None:
        self._config = config or get_settings()
        self._repository = DocumentRepository(
            registry_path or self._config.registry_file
        )
        self._engine_factory = engine_factory or (
            lambda doc_id: RAGEngine(
                collection_name=doc_id, config=self._config
            )
        )
        self._engines: dict[str, RAGEngine] = {}
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
                # 保留正常命名（与 doc_id 同名）及旧命名（doc_<doc_id>）的集合
                legacy_key = name[4:] if name.startswith("doc_") else name
                if name in self._documents or legacy_key in self._documents:
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

    def search(self, doc_id: str, query: str, top_k: int = 5) -> list[dict]:
        """在文档中检索。"""
        engine = self.get_engine(doc_id)
        if not engine:
            raise ValueError(f"文档不存在: {doc_id}")
        return engine.search(query, top_k)

    def delete_document(self, doc_id: str) -> None:
        """删除文档：移除索引集合、引擎实例与注册表记录。"""
        with self._lock:
            engine = self._engines.pop(doc_id, None)
            if engine is not None:
                engine.drop_index()
            self._documents.pop(doc_id, None)
            self._repository.save(self._documents)
