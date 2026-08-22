from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_knowledge_base
from src.core.config import settings
from src.core.models import DocumentInfo, DocumentStatus
from src.main import app


class FakeKnowledgeBase:
    """实现路由层用到的知识库服务接口。"""

    def __init__(self) -> None:
        self.docs: dict[str, DocumentInfo] = {}
        self._seq = 0

    def create_document(self, filename: str, file_size: int) -> str:
        self._seq += 1
        doc_id = f"doc_test_{self._seq}"
        now = datetime.now(UTC)
        self.docs[doc_id] = DocumentInfo(
            id=doc_id,
            filename=filename,
            file_size=file_size,
            status=DocumentStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        return doc_id

    def get_document_info(self, doc_id: str):
        return self.docs.get(doc_id)

    def list_documents(self):
        return list(self.docs.values())

    def update_document_file_size(self, doc_id: str, file_size: int) -> None:
        if doc_id in self.docs:
            self.docs[doc_id].file_size = file_size

    def get_engine(self, doc_id: str):
        if doc_id not in self.docs:
            return None

        class FakeEngine:
            def get_stats(self):
                return {
                    "collection": doc_id,
                    "total_vectors": 7,
                    "is_indexed": True,
                }

        return FakeEngine()

    def build_index(self, doc_id: str, pdf_path: str) -> dict:
        self.docs[doc_id].status = DocumentStatus.COMPLETED
        self.docs[doc_id].total_chunks = 3
        return {"total_chunks": 3}

    def search(self, doc_id: str, query: str, top_k: int = 5) -> list:
        return [
            {
                "text": "匹配内容",
                "type": "text",
                "page_num": 3,
                "block_id": 18,
                "score": 0.9,
            }
        ]

    def delete_document(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(settings, "max_upload_size_mb", 5)

    fake = FakeKnowledgeBase()
    app.dependency_overrides[get_knowledge_base] = lambda: fake
    with TestClient(app) as test_client:
        yield test_client, fake
    app.dependency_overrides.clear()


def test_health(client):
    test_client, _ = client
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_upload_rejects_non_pdf(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/upload",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_success_creates_doc_and_file(client, tmp_path):
    test_client, fake = client
    pdf_bytes = b"%PDF-1.4 fake content"
    response = test_client.post(
        "/api/v1/upload",
        files={"file": ("math.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "math.pdf"
    assert body["status"] == "pending"

    assert fake.get_document_info(body["id"]) is not None
    saved = tmp_path / "uploads" / f"{body['id']}.pdf"
    assert saved.read_bytes() == pdf_bytes


def test_upload_too_large(client):
    test_client, _ = client
    big_bytes = b"x" * (6 * 1024 * 1024)  # 6MB > 5MB 限制
    response = test_client.post(
        "/api/v1/upload",
        files={"file": ("big.pdf", big_bytes, "application/pdf")},
    )
    assert response.status_code == 413


def test_index_unknown_doc_404(client):
    test_client, _ = client
    response = test_client.post("/api/v1/index/not_exists")
    assert response.status_code == 404


def test_index_submits_task_and_completes(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    # 需要真实文件供路由检查
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    (upload_dir / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4")

    response = test_client.post(f"/api/v1/index/{doc_id}")
    assert response.status_code == 202
    assert response.json()["status"] == "processing"
    # TestClient 会同步执行 background tasks
    assert fake.get_document_info(doc_id).status == DocumentStatus.COMPLETED


def test_search_success(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    fake.docs[doc_id].status = DocumentStatus.COMPLETED

    response = test_client.post(
        f"/api/v1/search/{doc_id}", json={"query": "函数", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["results"][0]["text"] == "匹配内容"
    assert body["results"][0]["page_num"] == 3
    assert body["results"][0]["block_id"] == 18


def test_search_rejects_not_indexed(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    response = test_client.post(
        f"/api/v1/search/{doc_id}", json={"query": "函数"}
    )
    assert response.status_code == 400


def test_search_empty_query_422(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    fake.docs[doc_id].status = DocumentStatus.COMPLETED
    response = test_client.post(
        f"/api/v1/search/{doc_id}", json={"query": ""}
    )
    assert response.status_code == 422


def test_delete_document(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    response = test_client.delete(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    assert fake.get_document_info(doc_id) is None


def test_stats(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    response = test_client.get(f"/api/v1/stats/{doc_id}")
    assert response.status_code == 200
    assert response.json()["total_vectors"] == 7
