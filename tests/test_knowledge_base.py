import chromadb
import pytest

from src.core.config import Settings
from src.core.models import DocumentStatus
from src.services.knowledge_base import KnowledgeBaseService


class FakeEngine:
    """最小可用的 RAGEngine 替身。"""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.build_calls = 0
        self.dropped = False

    def build_index(self, pdf_path: str) -> dict:
        self.build_calls += 1
        return {"total_chunks": 3, "pages": 1, "collection": self.doc_id}

    def search(self, query: str, top_k: int = 5) -> list:
        return []

    @property
    def indexer(self):
        return self

    def drop(self) -> None:
        self.dropped = True

    def drop_index(self) -> None:
        self.dropped = True


class BoomEngine(FakeEngine):
    def build_index(self, pdf_path: str) -> dict:
        raise RuntimeError("boom")


def make_service(
    registry_path, engine_factory=None, reconcile=False
) -> KnowledgeBaseService:
    factory = engine_factory or (lambda doc_id: FakeEngine(doc_id))
    return KnowledgeBaseService(
        registry_path=registry_path,
        engine_factory=factory,
        reconcile=reconcile,
    )


def test_registry_persists_across_instances(tmp_path):
    path = tmp_path / "registry.json"
    service = make_service(path)
    doc_id = service.create_document("math.pdf", 123)

    service2 = make_service(path)
    doc = service2.get_document_info(doc_id)
    assert doc is not None
    assert doc.filename == "math.pdf"
    assert doc.file_size == 123
    assert doc.status == DocumentStatus.PENDING


def test_build_index_status_flow(tmp_path):
    path = tmp_path / "registry.json"
    service = make_service(path)
    doc_id = service.create_document("math.pdf", 123)

    service.build_index(doc_id, "/tmp/math.pdf")

    doc = service.get_document_info(doc_id)
    assert doc.status == DocumentStatus.COMPLETED
    assert doc.total_chunks == 3
    assert doc.error is None


def test_build_index_failure_marks_failed(tmp_path):
    path = tmp_path / "registry.json"
    service = make_service(path, engine_factory=lambda doc_id: BoomEngine(doc_id))
    doc_id = service.create_document("math.pdf", 123)

    with pytest.raises(RuntimeError, match="boom"):
        service.build_index(doc_id, "/tmp/math.pdf")

    doc = service.get_document_info(doc_id)
    assert doc.status == DocumentStatus.FAILED
    assert doc.error == "boom"


def test_delete_removes_registry_and_drops_index(tmp_path):
    path = tmp_path / "registry.json"
    service = make_service(path)
    doc_id = service.create_document("math.pdf", 123)
    engine = service.get_engine(doc_id)
    assert engine is not None

    service.delete_document(doc_id)

    assert service.get_document_info(doc_id) is None
    assert engine.dropped is True


def test_corrupted_registry_resets(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("{not valid json", encoding="utf-8")

    service = make_service(path)
    assert service.list_documents() == []


def test_cleanup_keeps_known_collections(tmp_path):
    config = Settings(
        _env_file=None,
        qwen_api_key="k",
        qwen_base_url="https://example.com/v1",
        paddleocr_api_key="p",
        chroma_persist_dir=str(tmp_path / "chroma"),
        registry_file=str(tmp_path / "registry.json"),
        upload_dir=str(tmp_path / "uploads"),
    )
    service = KnowledgeBaseService(
        config=config,
        reconcile=False,
        engine_factory=lambda doc_id: FakeEngine(doc_id),
    )
    doc_id = service.create_document("a.pdf", 1)

    client = chromadb.PersistentClient(
        path=config.chroma_persist_dir,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    client.create_collection(name=doc_id)  # 新命名：与 doc_id 同名
    client.create_collection(name=f"doc_{doc_id}")  # 旧命名：doc_<doc_id>
    client.create_collection(name="doc_orphan")  # 真正的孤儿

    service._cleanup_orphan_collections()

    names = [collection.name for collection in client.list_collections()]
    assert doc_id in names
    assert f"doc_{doc_id}" in names
    assert "doc_orphan" not in names
