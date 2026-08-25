"""检索评测指标：文本归一化、相关性判定、Recall/MRR/NDCG 与分层抽样。"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any

CATEGORIES = ("concept", "formula", "chapter", "no_answer")
EXPECTED_COUNTS = {"concept": 8, "formula": 8, "chapter": 7, "no_answer": 7}


def validate_dataset(items: list[dict[str, Any]]) -> None:
    """校验标注题集：id 唯一、分类合法、answer_terms 非空、类别计数符合预期。"""
    errors: list[str] = []
    seen: set[str] = set()
    counts = {cat: 0 for cat in CATEGORIES}
    for item in items:
        qid = item.get("id")
        if not qid:
            errors.append("存在缺少 id 的题目")
        elif qid in seen:
            errors.append(f"重复 id: {qid}")
        seen.add(qid)

        cat = item.get("category")
        if cat not in CATEGORIES:
            errors.append(f"{qid} 非法 category: {cat}")
        else:
            counts[cat] += 1

        terms = item.get("answer_terms")
        if (
            not isinstance(terms, list)
            or not terms
            or not all(isinstance(t, str) and t.strip() for t in terms)
        ):
            errors.append(f"{qid} answer_terms 必须是非空字符串列表")
        if "expected_absent" not in item:
            errors.append(f"{qid} 缺少 expected_absent")
        if not str(item.get("question", "")).strip():
            errors.append(f"{qid} question 为空")

    if counts != EXPECTED_COUNTS:
        errors.append(f"类别计数不符: {counts}，期望 {EXPECTED_COUNTS}")
    if errors:
        raise ValueError("评测集校验失败:\n" + "\n".join(errors))


def normalize_text(text: str) -> str:
    """归一化空白：去除首尾空白并把连续空白（含全角空格）折叠为单个空格。"""
    return " ".join(text.split())


def is_relevant(text: str, answer_terms: Iterable[str]) -> bool:
    """块文本归一化后是否包含任一答案术语（子串匹配）。"""
    norm = normalize_text(text)
    return any(
        normalize_text(term) in norm
        for term in answer_terms
        if term.strip()
    )


def hit_ranks(
    texts: list[str], answer_terms: list[str], top_k: int
) -> list[int]:
    """返回前 top_k 个结果中命中的 1-based 排名列表（按顺序）。"""
    return [
        i + 1
        for i, text in enumerate(texts[:top_k])
        if is_relevant(text, answer_terms)
    ]


def first_hit_rank(
    texts: list[str], answer_terms: list[str], top_k: int
) -> int | None:
    """首个相关结果的 1-based 排名；无命中返回 None。"""
    ranks = hit_ranks(texts, answer_terms, top_k)
    return ranks[0] if ranks else None


def recall_at_k(
    texts: list[str], answer_terms: list[str], top_k: int
) -> float:
    """单查询二元召回：前 k 个结果中有相关块为 1，否则 0。"""
    return 1.0 if first_hit_rank(texts, answer_terms, top_k) is not None else 0.0


def mrr_at_k(
    texts: list[str], answer_terms: list[str], top_k: int
) -> float:
    """单查询 MRR：1/首个相关排名，无命中为 0。"""
    rank = first_hit_rank(texts, answer_terms, top_k)
    return 1.0 / rank if rank else 0.0


def precision_at_k(
    texts: list[str], answer_terms: list[str], top_k: int
) -> float:
    """单查询 precision@k：前 k 个结果中相关块占比（无结果时为 0）。"""
    window = texts[:top_k]
    if not window:
        return 0.0
    return sum(1 for text in window if is_relevant(text, answer_terms)) / len(window)


def top1_relevant(
    texts: list[str], answer_terms: list[str]
) -> float:
    """top1 是否相关（0/1）。"""
    if not texts:
        return 0.0
    return 1.0 if is_relevant(texts[0], answer_terms) else 0.0


def ndcg_at_k(
    texts: list[str], answer_terms: list[str], top_k: int
) -> float:
    """单查询 NDCG@k（二元收益）：相关块收益 1，否则 0；无相关为 0。"""
    ranks = set(hit_ranks(texts, answer_terms, top_k))
    gains = [
        1.0 if i + 1 in ranks else 0.0
        for i in range(min(top_k, len(texts)))
    ]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def rejection_correct(
    texts: list[str], answer_terms: list[str], top_k: int
) -> bool:
    """无答案题：前 k 个结果无相关命中视为正确拒答。"""
    return first_hit_rank(texts, answer_terms, top_k) is None


def stratified_sample(
    items: list[dict[str, Any]], n: int, seed: int
) -> list[dict[str, Any]]:
    """按 category 分层抽样：每类至少 2 题（不足则全取），固定种子可复现。"""
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_cat.setdefault(item["category"], []).append(item)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    pools: dict[str, list[dict[str, Any]]] = {}
    for cat in sorted(by_cat):
        pool = list(by_cat[cat])
        rng.shuffle(pool)
        take = min(2, len(pool))
        selected.extend(pool[:take])
        pools[cat] = pool[take:]

    remaining = n - len(selected)
    cats = sorted(pools)
    idx = 0
    while remaining > 0 and any(pools.values()):
        cat = cats[idx % len(cats)]
        if pools[cat]:
            selected.append(pools[cat].pop(0))
            remaining -= 1
        idx += 1

    selected.sort(key=lambda it: it.get("id", ""))
    return selected
