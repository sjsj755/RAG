from src.core.fusion import reciprocal_rank_fusion


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
