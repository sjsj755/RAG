from src.core.models import ChunkResponse
from src.utils.helpers import ensure_dir, generate_document_id, to_dict


def test_to_dict_plain_objects():
    assert to_dict({"a": [1, 2], "b": {"c": "d"}}) == {
        "a": [1, 2],
        "b": {"c": "d"},
    }


def test_to_dict_pydantic_model():
    chunk = ChunkResponse(text="hi", page_num=2)
    assert to_dict(chunk) == {
        "text": "hi",
        "type": "text",
        "page_num": 2,
        "score": None,
        "block_id": None,
    }


def test_ensure_dir_creates_nested(tmp_path):
    path = ensure_dir(str(tmp_path / "a" / "b"))
    assert path.is_dir()


def test_generate_document_id_unique_and_formatted():
    id_a = generate_document_id("x.pdf")
    id_b = generate_document_id("x.pdf")
    assert id_a != id_b
    assert id_a.startswith("doc_")
    assert len(id_a.split("_")) == 3
