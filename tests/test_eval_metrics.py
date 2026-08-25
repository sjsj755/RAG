"""检索评测指标单元测试。"""

import json
from collections import Counter
from pathlib import Path

import pytest

from src.utils.eval_metrics import (
    first_hit_rank,
    hit_ranks,
    is_relevant,
    mrr_at_k,
    ndcg_at_k,
    normalize_text,
    precision_at_k,
    recall_at_k,
    rejection_correct,
    stratified_sample,
    top1_relevant,
    validate_dataset,
)


def test_normalize_text_collapses_whitespace():
    assert normalize_text(" a   b\u3000c ") == "a b c"
    assert normalize_text("") == ""


def test_is_relevant_matches_latex_term():
    text = "记作 $ a \\in A $"
    assert is_relevant(text, ["\\in A"])
    assert not is_relevant(text, ["\\notin A"])
    assert not is_relevant(text, ["完全无关"])


def test_hit_ranks_and_first_hit_rank():
    texts = ["甲", "乙", "丙 答案", "丁"]
    assert hit_ranks(texts, ["答案"], 5) == [3]
    assert first_hit_rank(texts, ["答案"], 5) == 3
    assert first_hit_rank(texts, ["没有"], 5) is None
    assert first_hit_rank(texts, ["答案"], 2) is None


def test_recall_and_mrr():
    texts = ["a", "b", "答案", "d", "e"]
    assert recall_at_k(texts, ["答案"], 5) == 1.0
    assert mrr_at_k(texts, ["答案"], 5) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "b", "c"], ["答案"], 5) == 0.0
    assert mrr_at_k(["a", "b", "c"], ["答案"], 5) == 0.0


def test_precision_at_k():
    texts = ["答案", "a", "答案", "b", "c"]
    assert precision_at_k(texts, ["答案"], 5) == 0.4
    assert precision_at_k(["a", "b"], ["答案"], 5) == 0.0
    assert precision_at_k([], ["答案"], 5) == 0.0


def test_top1_relevant():
    assert top1_relevant(["答案", "b"], ["答案"]) == 1.0
    assert top1_relevant(["a", "答案"], ["答案"]) == 0.0
    assert top1_relevant([], ["答案"]) == 0.0


def test_ndcg_binary_gains():
    texts = ["a", "b", "答案", "d", "e"]
    # 相关在 rank3：DCG=1/log2(4)=0.5，IDCG=1/log2(2)=1
    assert ndcg_at_k(texts, ["答案"], 5) == pytest.approx(0.5)
    # 相关在 rank1
    assert ndcg_at_k(["答案", "b", "c"], ["答案"], 5) == pytest.approx(1.0)
    # 无相关
    assert ndcg_at_k(["a", "b", "c", "d", "e"], ["答案"], 5) == 0.0


def test_rejection_correct_for_no_answer():
    assert rejection_correct(["a", "b"], ["答案"], 5) is True
    assert rejection_correct(["a", "答案"], ["答案"], 5) is False


def test_stratified_sample_balanced_and_reproducible():
    items = []
    for i, cat in enumerate(
        ["concept"] * 8
        + ["formula"] * 8
        + ["chapter"] * 7
        + ["no_answer"] * 7
    ):
        items.append({"id": f"q{i:02d}", "category": cat})

    sample1 = stratified_sample(items, 10, 42)
    sample2 = stratified_sample(items, 10, 42)

    assert [it["id"] for it in sample1] == [it["id"] for it in sample2]
    assert len(sample1) == 10
    counts = Counter(it["category"] for it in sample1)
    for cat in ("concept", "formula", "chapter", "no_answer"):
        assert counts[cat] >= 2


def test_validate_dataset_accepts_repo_dataset():
    items = json.loads(
        Path("scripts/eval_questions.json").read_text(encoding="utf-8")
    )
    validate_dataset(items)


def test_validate_dataset_rejects_bad_items():
    base = {
        "id": "q01",
        "category": "concept",
        "question": "问题？",
        "answer_terms": ["答案"],
        "expected_absent": False,
    }

    with pytest.raises(ValueError, match="重复 id"):
        validate_dataset([base, {**base}])

    with pytest.raises(ValueError, match="非法 category"):
        validate_dataset([{**base, "category": "unknown"}])

    with pytest.raises(ValueError, match="answer_terms"):
        validate_dataset([{**base, "answer_terms": []}])

    with pytest.raises(ValueError, match="类别计数不符"):
        validate_dataset([base])
