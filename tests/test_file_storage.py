import asyncio

import pytest

from src.services.document_repository import DocumentRepository
from src.services.file_storage import FileStorage, FileTooLargeError


def test_directory_created(tmp_path):
    storage = FileStorage(tmp_path / "a" / "b")
    assert storage.directory.is_dir()


def test_path_for_and_exists(tmp_path):
    storage = FileStorage(tmp_path / "uploads")
    assert storage.path_for("doc_1") == tmp_path / "uploads" / "doc_1.pdf"
    assert not storage.exists("doc_1")
    storage.path_for("doc_1").write_bytes(b"x")
    assert storage.exists("doc_1")


def test_delete_is_idempotent(tmp_path):
    storage = FileStorage(tmp_path / "uploads")
    storage.path_for("doc_1").write_bytes(b"x")
    storage.delete("doc_1")
    assert not storage.exists("doc_1")
    storage.delete("doc_1")  # 不存在时静默


def test_save_writes_atomically(tmp_path):
    storage = FileStorage(tmp_path / "uploads")
    chunks = iter([b"abc", b"def", b""])

    async def reader() -> bytes:
        return next(chunks)

    written = asyncio.run(storage.save("doc_1", reader, 1024))
    assert written == 6
    assert storage.path_for("doc_1").read_bytes() == b"abcdef"
    assert not storage.path_for("doc_1").with_suffix(".pdf.tmp").exists()


def test_save_raises_when_too_large(tmp_path):
    storage = FileStorage(tmp_path / "uploads")
    chunks = iter([b"x" * 5, b""])

    async def reader() -> bytes:
        return next(chunks)

    with pytest.raises(FileTooLargeError):
        asyncio.run(storage.save("doc_1", reader, 3))
    assert not storage.path_for("doc_1").exists()
    assert not storage.path_for("doc_1").with_suffix(".pdf.tmp").exists()


def test_repository_roundtrip(tmp_path):
    from datetime import UTC, datetime

    from src.core.models import DocumentInfo, DocumentStatus

    repo = DocumentRepository(tmp_path / "registry.json")
    now = datetime.now(UTC)
    doc = DocumentInfo(
        id="doc_1",
        filename="a.pdf",
        file_size=1,
        status=DocumentStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    repo.save({"doc_1": doc})

    loaded = repo.load()
    assert loaded["doc_1"].filename == "a.pdf"
    assert loaded["doc_1"].status == DocumentStatus.PENDING


def test_repository_corrupted_resets(tmp_path):
    path = tmp_path / "registry.json"
    path.write_text("{bad json", encoding="utf-8")
    assert DocumentRepository(path).load() == {}
