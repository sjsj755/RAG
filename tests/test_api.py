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
        self.refused = False
        self.answer_error = False

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

    def search_with_confidence(
        self, doc_id: str, query: str, top_k: int = 5
    ) -> dict:
        return {
            "results": self.search(doc_id, query, top_k),
            "confidence": 0.9,
            "refused": False,
        }

    def answer(self, doc_id: str, query: str, top_k: int = 5) -> dict:
        if self.answer_error:
            raise RuntimeError("boom")
        if self.refused:
            return {
                "query": query,
                "answer": None,
                "refused": True,
                "refusal_reason": "检索置信度不足，未生成答案",
                "confidence": 0.3,
                "sources": [],
            }
        return {
            "query": query,
            "answer": "集合是某些对象的总体，定义答案",
            "refused": False,
            "refusal_reason": None,
            "confidence": 0.9,
            "sources": [
                {"index": 1, "page_num": 3, "text": "片段", "score": 0.9}
            ],
        }

    def stream_answer(self, doc_id: str, query: str, top_k: int = 5):
        if self.answer_error:
            raise RuntimeError("boom")
        if self.refused:
            yield {
                "type": "refused",
                "reason": "检索置信度不足，未生成答案",
                "confidence": 0.3,
            }
            yield {"type": "done", "refused": True, "confidence": 0.3}
            return
        yield {
            "type": "sources",
            "sources": [
                {"index": 1, "page_num": 3, "text": "片段", "score": 0.9}
            ],
            "confidence": 0.9,
        }
        yield {"type": "answer", "content": "定义"}
        yield {"type": "answer", "content": "答案"}
        yield {"type": "done", "refused": False, "confidence": 0.9}

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


def test_upload_rejects_fake_pdf(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/upload",
        files={"file": ("fake.pdf", b"not a real pdf", "application/pdf")},
    )
    assert response.status_code == 400
    assert "不是有效的 PDF 文件" in response.json()["detail"]


def test_upload_rejects_empty_file(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "文件为空" in response.json()["detail"]


def test_upload_too_large(client):
    test_client, _ = client
    big_bytes = b"%PDF-1.4 " + b"x" * (6 * 1024 * 1024)  # 合法头 + 6MB > 5MB
    response = test_client.post(
        "/api/v1/upload",
        files={"file": ("big.pdf", big_bytes, "application/pdf")},
    )
    assert response.status_code == 413


def test_batch_upload_success(client):
    test_client, fake = client
    pdf_bytes = b"%PDF-1.4 fake content"
    response = test_client.post(
        "/api/v1/upload/batch",
        files=[
            ("files", ("math1.pdf", pdf_bytes, "application/pdf")),
            ("files", ("math2.pdf", pdf_bytes, "application/pdf")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert body["succeeded"] == 2
    assert body["failed"] == 0
    assert all(item["status"] == "uploaded" for item in body["results"])
    assert all(item["doc_id"] for item in body["results"])
    assert len(fake.docs) == 2


def test_batch_upload_partial_failure_isolated(client):
    test_client, fake = client
    response = test_client.post(
        "/api/v1/upload/batch",
        files=[
            ("files", ("good.pdf", b"%PDF-1.4 ok", "application/pdf")),
            ("files", ("notes.txt", b"hello", "text/plain")),
            ("files", ("fake.pdf", b"not a pdf", "application/pdf")),
            ("files", ("empty.pdf", b"", "application/pdf")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert body["succeeded"] == 1
    assert body["failed"] == 3

    by_name = {item["filename"]: item for item in body["results"]}
    assert by_name["good.pdf"]["status"] == "uploaded"
    assert by_name["notes.txt"]["status"] == "rejected"
    assert "仅支持 PDF 文件" in by_name["notes.txt"]["error"]
    assert "不是有效的 PDF 文件" in by_name["fake.pdf"]["error"]
    assert "文件为空" in by_name["empty.pdf"]["error"]

    # 失败项不残留注册表记录
    assert len(fake.docs) == 1


def test_batch_upload_too_many_files(client, monkeypatch):
    test_client, _ = client
    monkeypatch.setattr(settings, "max_batch_files", 2)
    pdf_bytes = b"%PDF-1.4 fake"
    response = test_client.post(
        "/api/v1/upload/batch",
        files=[
            ("files", (f"m{i}.pdf", pdf_bytes, "application/pdf"))
            for i in range(3)
        ],
    )
    assert response.status_code == 400
    assert "一次最多上传 2 个文件" in response.json()["detail"]


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


def test_answer_success(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    fake.docs[doc_id].status = DocumentStatus.COMPLETED

    response = test_client.post(
        f"/api/v1/answer/{doc_id}",
        json={"query": "什么是集合？", "top_k": 3},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "集合是某些对象的总体，定义答案"
    assert body["refused"] is False
    assert body["confidence"] == 0.9
    assert body["sources"][0]["index"] == 1
    assert body["sources"][0]["page_num"] == 3


def test_answer_refused(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    fake.docs[doc_id].status = DocumentStatus.COMPLETED
    fake.refused = True

    response = test_client.post(
        f"/api/v1/answer/{doc_id}", json={"query": "量子纠缠是什么？"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["answer"] is None
    assert body["sources"] == []
    assert "检索置信度不足" in body["refusal_reason"]


def test_answer_stream(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    fake.docs[doc_id].status = DocumentStatus.COMPLETED

    response = test_client.post(
        f"/api/v1/answer/{doc_id}?stream=true",
        json={"query": "什么是集合？"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type":"sources"' in body
    assert '"type":"answer"' in body
    assert '"type":"done"' in body


def test_answer_stream_refused(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    fake.docs[doc_id].status = DocumentStatus.COMPLETED
    fake.refused = True

    response = test_client.post(
        f"/api/v1/answer/{doc_id}?stream=true",
        json={"query": "量子纠缠是什么？"},
    )
    assert response.status_code == 200
    body = response.text
    assert '"type":"refused"' in body
    assert '"type":"done"' in body
    assert '"type":"answer"' not in body


def test_answer_unknown_doc_404(client):
    test_client, _ = client
    response = test_client.post(
        "/api/v1/answer/not_exists", json={"query": "什么是集合？"}
    )
    assert response.status_code == 404


def test_answer_not_indexed_400(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    response = test_client.post(
        f"/api/v1/answer/{doc_id}", json={"query": "什么是集合？"}
    )
    assert response.status_code == 400


def test_answer_generation_failure_502(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    fake.docs[doc_id].status = DocumentStatus.COMPLETED
    fake.answer_error = True

    response = test_client.post(
        f"/api/v1/answer/{doc_id}", json={"query": "什么是集合？"}
    )
    assert response.status_code == 502


def test_answer_stream_error_event(client):
    test_client, fake = client
    doc_id = fake.create_document("math.pdf", 10)
    fake.docs[doc_id].status = DocumentStatus.COMPLETED
    fake.answer_error = True

    response = test_client.post(
        f"/api/v1/answer/{doc_id}?stream=true",
        json={"query": "什么是集合？"},
    )
    assert response.status_code == 200
    assert '"type":"error"' in response.text
