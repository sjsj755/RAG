"""BM25 稀疏检索路：精确词匹配，作为向量检索的互补路。"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.utils.math_terms import MATH_TERMS

_ASCII_TOKEN = re.compile(r"[A-Za-z0-9_\\{}^+*/=\-()\[\].]+")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_TERMS_BY_LENGTH = sorted(MATH_TERMS, key=len, reverse=True)


def tokenize(text: str) -> list[str]:
    """轻量中文分词：KG 术语最长匹配 + 中文双字 bigram + 连续 ASCII/LaTeX token。"""
    tokens: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for term in _TERMS_BY_LENGTH:
            if text.startswith(term, i):
                tokens.append(term)
                i += len(term)
                matched = True
                break
        if matched:
            continue

        ch = text[i]
        ascii_match = _ASCII_TOKEN.match(text, i)
        if ascii_match:
            tokens.append(ascii_match.group())
            i = ascii_match.end()
            continue
        if _CJK.match(ch):
            bigram = text[i : i + 2]
            if len(bigram) == 2 and _CJK.match(bigram[1]):
                tokens.append(bigram)
                i += 1
            else:
                tokens.append(ch)
                i += 1
            continue
        i += 1
    return tokens


class BM25Retriever:
    """BM25Okapi 封装：构建语料并返回候选块。"""

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict[str, Any]] = []

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """从内容块构建 BM25 索引。"""
        self._chunks = list(chunks)
        tokenized = [tokenize(c.get("text", "")) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if self._chunks else None

    def query(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        """返回 top_k 个非零分候选（按 BM25 得分降序）。"""
        if self._bm25 is None or not self._chunks:
            return []
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        order = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )
        results: list[dict[str, Any]] = []
        for i in order[:top_k]:
            if scores[i] <= 0:
                break
            item = dict(self._chunks[i])
            item["score"] = float(scores[i])
            results.append(item)
        return results
