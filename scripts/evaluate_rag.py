"""RAG 检索评测脚本。

用法:
    python scripts/evaluate_rag.py --questions scripts/eval_questions.json \
        --doc-id <doc_id> --top-k 5 --label rewrite \
        --output eval_report_rewrite.json --sample-human 10 --seed 42

不带 --questions 时沿用旧版 9 题（仅输出结果供人工评分，保持兼容）。
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

from src.utils.eval_metrics import (
    CATEGORIES,
    first_hit_rank,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    rejection_correct,
    stratified_sample,
    validate_dataset,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_DOC_ID = "doc_20260822142017_6f555236"
DEFAULT_TOP_K = 5

# 旧版 9 题：无 answer_terms，仅输出结果供人工评分（保持兼容）
QUESTIONS = [
    {
        "id": "q1",
        "question": "什么是集合？集合中的元素具有哪些基本性质？",
        "expected": "集合与元素定义；元素具有确定性和互异性",
    },
    {
        "id": "q2",
        "question": "如何用符号表示一个元素属于某个集合？举一个例子。",
        "expected": "a∈A 表示元素a属于集合A；例：4∈A",
    },
    {
        "id": "q3",
        "question": "常用的数集有哪些？分别用什么字母表示？",
        "expected": "N非负整数、N*或N+正整数、Z整数、Q有理数、R实数",
    },
    {
        "id": "q4",
        "question": "什么是列举法？如何用列举法表示一个集合？",
        "expected": "把集合的所有元素一一列举出来，用花括号括起来",
    },
    {
        "id": "q5",
        "question": "方程 x^2 = x 的所有实数根组成的集合是什么？",
        "expected": "B={0,1}",
    },
    {
        "id": "q6",
        "question": "用列举法表示：小于10的所有自然数组成的集合。",
        "expected": "A={0,1,2,3,4,5,6,7,8,9}",
    },
    {
        "id": "q7",
        "question": "为什么身材较高的人不能构成一个集合？",
        "expected": "元素不确定，不满足集合的确定性",
    },
    {
        "id": "q8",
        "question": "人教A版必修一教材主要包含哪些章节内容？",
        "expected": (
            "集合与常用逻辑用语、一元二次函数方程和不等式、"
            "函数的概念与性质、指数函数与对数函数、三角函数"
        ),
    },
    {
        "id": "q9",
        "question": "集合的交集与并集运算规则是什么？",
        "expected": "文档提及1.3集合的基本运算，但前10页无具体内容（部分覆盖/陷阱题）",
    },
]


def load_questions(path: str | None) -> list[dict]:
    """加载标注题集；未指定时返回旧版 9 题（兼容）。"""
    if not path:
        return [dict(q) for q in QUESTIONS]
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_dataset(items)
    return items


def fetch_results(
    client: httpx.Client,
    base_url: str,
    doc_id: str,
    query: str,
    top_k: int,
) -> dict:
    resp = client.post(
        f"{base_url}/api/v1/search/{doc_id}",
        json={"query": query, "top_k": top_k},
    )
    resp.raise_for_status()
    return resp.json()


def compute_item(item: dict, payload: dict, top_k: int) -> dict:
    """计算单题指标；旧版题（无 answer_terms）仅返回结果。"""
    results = payload.get("results", [])
    refused = bool(payload.get("refused", False))
    out = {
        "id": item["id"],
        "category": item.get("category"),
        "question": item["question"],
        "expected": item.get("expected"),
        "answer_terms": item.get("answer_terms"),
        "expected_absent": bool(item.get("expected_absent", False)),
        "note": item.get("note", ""),
        "refused": refused,
        "confidence": payload.get("confidence"),
        "results": results,
    }
    terms = item.get("answer_terms")
    if not terms:
        return out

    texts = [r.get("text", "") or "" for r in results]
    rank = first_hit_rank(texts, terms, top_k)
    out["hit_rank"] = rank
    out["relevant"] = rank is not None
    # 被拒绝时视为 miss（即使结果列表里恰好有命中块）
    out[f"recall_at_{top_k}"] = (
        0.0 if refused else recall_at_k(texts, terms, top_k)
    )
    out[f"mrr_at_{top_k}"] = 0.0 if refused else mrr_at_k(texts, terms, top_k)
    out[f"ndcg_at_{top_k}"] = 0.0 if refused else ndcg_at_k(texts, terms, top_k)
    out["rejected_correctly"] = (
        refused or rejection_correct(texts, terms, top_k)
        if out["expected_absent"]
        else None
    )
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(items: list[dict], top_k: int) -> dict:
    """汇总指标：有答案题算 Recall/MRR/NDCG，无答案题单独算拒答正确率。"""
    metric_keys = (f"recall_at_{top_k}", f"mrr_at_{top_k}", f"ndcg_at_{top_k}")
    answerable = [
        it for it in items if it.get("answer_terms") and not it["expected_absent"]
    ]
    no_answer = [
        it for it in items if it.get("answer_terms") and it["expected_absent"]
    ]

    summary = {
        "answerable_count": len(answerable),
        "no_answer_count": len(no_answer),
        "rejection_rate": _mean(
            [1.0 if it["rejected_correctly"] else 0.0 for it in no_answer]
        ),
        "false_refusal_rate": _mean(
            [1.0 if it["refused"] else 0.0 for it in answerable]
        ),
        "by_category": {},
    }
    for key in metric_keys:
        summary[key] = _mean([it[key] for it in answerable])

    for cat in CATEGORIES:
        group = [it for it in items if it.get("category") == cat]
        if not group or not group[0].get("answer_terms"):
            continue
        if cat == "no_answer":
            summary["by_category"][cat] = {
                "count": len(group),
                "rejection_rate": _mean(
                    [1.0 if it["rejected_correctly"] else 0.0 for it in group]
                ),
            }
        else:
            summary["by_category"][cat] = {
                "count": len(group),
                **{
                    key: _mean([it[key] for it in group])
                    for key in metric_keys
                },
            }
    return summary


def build_human_review(items: list[dict], count: int, seed: int) -> dict:
    """分层抽样生成人工复核文件（每类至少 2 题，固定种子可复现）。"""
    sample = stratified_sample(items, count, seed)
    return {
        "seed": seed,
        "count": len(sample),
        "instructions": (
            "每题按 1-5 打分：5=top1 完全回答；4=top1-2 回答；"
            "3=top3 回答；2=top4-5 部分回答；1=未召回或无答案题误召回。"
            "将 score 填写为数字后保存。"
        ),
        "items": [
            {
                "id": it["id"],
                "category": it["category"],
                "question": it["question"],
                "answer_terms": it.get("answer_terms"),
                "note": it.get("note", ""),
                "expected_absent": it.get("expected_absent", False),
                "auto_hit_rank": it.get("hit_rank"),
                "auto_relevant": it.get("relevant"),
                "auto_refused": it.get("refused"),
                "confidence": it.get("confidence"),
                "results": [
                    {
                        "page_num": r.get("page_num"),
                        "type": r.get("type"),
                        "text": (r.get("text") or "")[:180],
                    }
                    for r in it["results"]
                ],
                "score": None,
            }
            for it in sample
        ],
    }


def run_evaluation(
    base_url: str,
    doc_id: str,
    top_k: int,
    questions: list[dict],
    label: str,
) -> dict:
    items: list[dict] = []
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        for question in questions:
            results = fetch_results(
                client, base_url, doc_id, question["question"], top_k
            )
            items.append(compute_item(question, results, top_k))

    report = {
        "label": label,
        "doc_id": doc_id,
        "top_k": top_k,
        "generated_at": datetime.now(UTC).isoformat(),
        "items": items,
    }
    if all(it.get("answer_terms") for it in questions):
        report["summary"] = summarize(items, top_k)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--questions", help="评测题集 JSON 路径")
    parser.add_argument("--label", default="eval", help="报告标签，如 rewrite/no_rewrite")
    parser.add_argument("--output", default="eval_report.json")
    parser.add_argument(
        "--sample-human",
        type=int,
        default=0,
        help="抽样 N 题生成人工复核文件",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    questions = load_questions(args.questions)
    report = run_evaluation(
        args.base_url, args.doc_id, args.top_k, questions, args.label
    )
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"评测完成：{len(report['items'])} 个问题，结果已写入 {args.output}")

    if args.sample_human > 0:
        review = build_human_review(report["items"], args.sample_human, args.seed)
        review_path = args.output.replace("eval_report", "eval_human_review")
        Path(review_path).write_text(
            json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"人工复核文件已写入 {review_path}")


if __name__ == "__main__":
    main()
