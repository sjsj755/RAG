from src.core.fusion import item_key, reciprocal_rank_fusion


def _item(text: str, page: int = 1) -> dict:
    return {"text": text, "page_num": page, "block_id": "b", "type": "text"}


def test_single_list_keeps_order():
    items = [_item("a"), _item("b"), _item("c")]
    assert reciprocal_rank_fusion([items]) == items


def test_merge_two_lists_and_deduplicate():
    list_a = [_item("a"), _item("b"), _item("c")]
    list_b = [_item("x"), _item("a"), _item("y")]
    merged = reciprocal_rank_fusion([list_a, list_b], k=60)

    texts = [item["text"] for item in merged]
    assert texts[0] == "a"  # 两路都排第 1/2 位，RRF 分最高
    assert texts.count("a") == 1
    assert len(merged) == 5


def test_top_k_limit():
    items = [_item(f"t{i}") for i in range(10)]
    assert len(reciprocal_rank_fusion([items], top_k=3)) == 3


def test_deduplicate_by_parent_id():
    list_a = [
        {
            "text": "片段A1",
            "page_num": 1,
            "block_id": "b1",
            "parent_id": "parent_1",
        }
    ]
    list_b = [
        {
            "text": "片段A2",
            "page_num": 1,
            "block_id": "b1",
            "parent_id": "parent_1",
        }
    ]
    merged = reciprocal_rank_fusion([list_a, list_b])
    assert len(merged) == 1
    assert merged[0]["parent_id"] == "parent_1"


def test_item_key_priority():
    with_parent = {"parent_id": "p1", "page_num": 2, "text": "x"}
    assert item_key(with_parent) == ("parent", "p1", 2)
    without_parent = {"page_num": 2, "block_id": "b", "text": "x"}
    assert item_key(without_parent)[0] == 2
