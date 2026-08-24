"""多路召回融合：Reciprocal Rank Fusion（RRF）。"""

from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """按排名位置融合多路候选，去重后返回排序结果。"""
    scores: dict[tuple, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = (
                item.get("page_num", 0),
                str(item.get("block_id", "")),
                item.get("text", "")[:200],
            )
            entry = scores.setdefault(key, {"item": item, "score": 0.0})
            entry["score"] += 1.0 / (k + rank)

    merged = sorted(
        scores.values(), key=lambda entry: entry["score"], reverse=True
    )
    items = [entry["item"] for entry in merged]
    return items[:top_k] if top_k else items
