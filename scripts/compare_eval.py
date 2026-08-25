"""合并两组评测报告（如 rewrite vs no_rewrite），输出逐题与汇总对比。

用法:
    python scripts/compare_eval.py \
        --a eval_report_rewrite.json --b eval_report_no_rewrite.json \
        --human-a eval_human_review_rewrite.json \
        --human-b eval_human_review_no_rewrite.json \
        --output eval_report.json
"""

import argparse
import json
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_report(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def human_summary(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scores = [
        it.get("score")
        for it in data.get("items", [])
        if isinstance(it.get("score"), (int, float))
    ]
    if not scores:
        return {"filled": 0, "average": None}
    return {"filled": len(scores), "average": round(_mean(scores), 4)}


def _pick(item: dict | None) -> dict | None:
    if not item:
        return None
    return {
        "hit_rank": item.get("hit_rank"),
        "relevant": item.get("relevant"),
        "rejected_correctly": item.get("rejected_correctly"),
        "refused": item.get("refused"),
        "confidence": item.get("confidence"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="合并两组评测报告")
    parser.add_argument("--a", required=True, help="A 组报告")
    parser.add_argument("--b", required=True, help="B 组报告")
    parser.add_argument("--output", default="eval_report.json")
    parser.add_argument("--human-a", help="A 组人工复核文件（可选）")
    parser.add_argument("--human-b", help="B 组人工复核文件（可选）")
    args = parser.parse_args()

    a = load_report(args.a)
    b = load_report(args.b)
    k = a.get("top_k", b.get("top_k", 5))
    metric_names = [
        f"recall_at_{k}",
        f"mrr_at_{k}",
        f"ndcg_at_{k}",
        "rejection_rate",
    ]

    sa = a.get("summary", {})
    sb = b.get("summary", {})
    summary = {}
    for name in metric_names:
        if name in sa and name in sb:
            summary[name] = {
                "a": sa[name],
                "b": sb[name],
                "delta": round(sb[name] - sa[name], 4),
            }

    by_category = {}
    for cat in sorted(set(sa.get("by_category", {})) | set(sb.get("by_category", {}))):
        ca = sa.get("by_category", {}).get(cat, {})
        cb = sb.get("by_category", {}).get(cat, {})
        entry: dict = {"a": ca, "b": cb}
        for name in metric_names:
            if name in ca and name in cb:
                entry[f"{name}_delta"] = round(cb[name] - ca[name], 4)
        by_category[cat] = entry

    items_a = {it["id"]: it for it in a.get("items", [])}
    items_b = {it["id"]: it for it in b.get("items", [])}
    merged_items = []
    for qid in sorted(set(items_a) | set(items_b)):
        ia = items_a.get(qid)
        ib = items_b.get(qid)
        base = ia or ib
        entry = {
            "id": qid,
            "category": base.get("category"),
            "question": base.get("question"),
            "note": base.get("note", ""),
            "expected_absent": base.get("expected_absent", False),
            "a": _pick(ia),
            "b": _pick(ib),
        }
        for name in metric_names[:3]:
            va = ia.get(name) if ia else None
            vb = ib.get(name) if ib else None
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                entry[f"{name}_delta"] = round(vb - va, 4)
        merged_items.append(entry)

    report = {
        "label_a": a.get("label", "a"),
        "label_b": b.get("label", "b"),
        "doc_id": a.get("doc_id"),
        "top_k": k,
        "summary": summary,
        "by_category": by_category,
        "items": merged_items,
    }
    if args.human_a or args.human_b:
        report["human"] = {}
        if args.human_a:
            report["human"][a.get("label", "a")] = human_summary(args.human_a)
        if args.human_b:
            report["human"][b.get("label", "b")] = human_summary(args.human_b)

    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"汇总对比（{a.get('label')} vs {b.get('label')}）:")
    for name, entry in summary.items():
        print(
            f"  {name}: {entry['a']} -> {entry['b']} "
            f"(Δ {entry['delta']:+.4f})"
        )
    for cat, entry in by_category.items():
        print(f"  [{cat}] {entry}")
    if "human" in report:
        print(f"  人工平均分: {report['human']}")
    print(f"对比报告已写入 {args.output}")


if __name__ == "__main__":
    main()
