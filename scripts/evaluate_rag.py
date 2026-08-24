"""RAG 检索评测脚本。

对预设问题集调用检索 API，输出 top-k 结果到 JSON 文件（UTF-8），
供人工按相关性评分。

用法:
    python scripts/evaluate_rag.py --output eval_results.json
"""

import argparse
import json

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_DOC_ID = "doc_20260822142017_6f555236"
DEFAULT_TOP_K = 5

# 问题集：基于人教A版必修一（前10页：主编寄语/目录/1.1集合的概念）设计
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


def run_evaluation(base_url: str, doc_id: str, top_k: int) -> dict:
    """对每个问题执行检索，返回结构化结果。"""
    items = []
    with httpx.Client(base_url=base_url, timeout=60.0) as client:
        for question in QUESTIONS:
            resp = client.post(
                f"/api/v1/search/{doc_id}",
                json={"query": question["question"], "top_k": top_k},
            )
            resp.raise_for_status()
            payload = resp.json()
            items.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "expected": question["expected"],
                    "results": payload["results"],
                }
            )
    return {"doc_id": doc_id, "top_k": top_k, "items": items}


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 检索评测")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--doc-id", default=DEFAULT_DOC_ID)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--output", default="eval_results.json", help="结果输出文件（UTF-8 JSON）"
    )
    args = parser.parse_args()

    report = run_evaluation(args.base_url, args.doc_id, args.top_k)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"评测完成：{len(report['items'])} 个问题，结果已写入 {args.output}")


if __name__ == "__main__":
    main()
