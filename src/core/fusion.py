"""多路召回融合：Reciprocal Rank Fusion（RRF）。"""

from typing import Any


def item_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """融合去重键：优先按父块+页码+块号，缺失时回退页码+块号+文本前缀。"""
    parent_id = item.get("parent_id")
    if parent_id:
        return (
            "parent",
            str(parent_id),
            item.get("page_num", 0),
            str(item.get("block_id", "")),
        )
    return (
        item.get("page_num", 0),
        str(item.get("block_id", "")),
        item.get("text", "")[:200],
    )


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """按排名位置融合多路候选，去重后返回排序结果。"""
    scores: dict[tuple, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = item_key(item)
            entry = scores.setdefault(key, {"item": item, "score": 0.0})
            entry["score"] += 1.0 / (k + rank)

    merged = sorted(
        scores.values(), key=lambda entry: entry["score"], reverse=True
    )
    items = [entry["item"] for entry in merged]
    return items[:top_k] if top_k else items
