from src.core.models import ChunkResponse
from src.utils.helpers import (
    compute_confidence,
    ensure_dir,
    generate_document_id,
    to_dict,
)


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
        "fragment": None,
        "doc_id": None,
        "filename": None,
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


def test_compute_confidence_single_leg():
    assert compute_confidence(0.7, 1, 1) == 0.7


def test_compute_confidence_multi_leg_agree():
    assert compute_confidence(0.8, 2, 2) == 0.8


def test_compute_confidence_multi_leg_disagree():
    # 3 路中只有 1 路命中 top1：0.8 × (0.5 + 0.5/3)
    assert compute_confidence(0.8, 1, 3) == 0.8 * (0.5 + 0.5 / 3)


def test_compute_confidence_no_vector_or_no_legs():
    assert compute_confidence(0.0, 1, 1) == 0.0
    assert compute_confidence(0.7, 0, 0) == 0.0
