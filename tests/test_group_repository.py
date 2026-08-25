"""分组注册表持久化单元测试。"""

from datetime import UTC, datetime

from src.core.models import GroupInfo
from src.services.group_repository import GroupRepository


def _group(group_id: str = "group_1", name: str = "必修一") -> GroupInfo:
    now = datetime.now(UTC)
    return GroupInfo(
        id=group_id,
        name=name,
        doc_ids=["doc_a", "doc_b"],
        created_at=now,
        updated_at=now,
    )


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "groups.json"
    repo = GroupRepository(path)
    repo.save({"group_1": _group()})

    loaded = GroupRepository(path).load()
    assert loaded["group_1"].name == "必修一"
    assert loaded["group_1"].doc_ids == ["doc_a", "doc_b"]


def test_load_missing_file_returns_empty(tmp_path):
    assert GroupRepository(tmp_path / "missing.json").load() == {}


def test_load_corrupted_file_resets(tmp_path):
    path = tmp_path / "groups.json"
    path.write_text("{not valid", encoding="utf-8")
    assert GroupRepository(path).load() == {}


def test_atomic_save_replaces_file(tmp_path):
    path = tmp_path / "groups.json"
    repo = GroupRepository(path)
    repo.save({"group_1": _group()})
    repo.save({"group_2": _group("group_2", "必修二")})
    data = GroupRepository(path).load()
    assert list(data) == ["group_2"]
