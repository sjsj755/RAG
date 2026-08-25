"""轻量知识图谱：规则抽取概念-块映射，JSON 持久化。"""

import json
from pathlib import Path
from typing import Any

from src.utils.logger import logger
from src.utils.math_terms import MATH_TERMS

_TERMS_BY_LENGTH = sorted(MATH_TERMS, key=len, reverse=True)

_TITLE_TYPES = {"doc_title", "paragraph_title"}


class RuleKnowledgeGraph:
    """规则抽取的知识图谱：概念→块、概念共现、标题→块映射。"""

    def __init__(self) -> None:
        self._chunks: dict[str, dict[str, Any]] = {}
        self._concept_to_chunks: dict[str, set[str]] = {}
        self._chunk_to_concepts: dict[str, set[str]] = {}
        self._cooccur: dict[str, set[str]] = {}
        self._titles: list[str] = []
        self._title_to_chunks: dict[str, set[str]] = {}

    # ---------- 构建 ----------

    def build(self, chunks: list[dict[str, Any]]) -> None:
        """从分块结果抽取概念与关系。"""
        self._chunks = {f"c_{i}": chunk for i, chunk in enumerate(chunks)}
        self._concept_to_chunks = {}
        self._chunk_to_concepts = {}
        self._cooccur = {}
        self._titles = []
        self._title_to_chunks = {}

        for chunk_id, chunk in self._chunks.items():
            text = chunk.get("text", "")
            concepts = self._match_terms(text)

            if chunk.get("type") in _TITLE_TYPES:
                title = text.strip()
                if title:
                    self._titles.append(title)
                    self._title_to_chunks.setdefault(title, set()).add(chunk_id)

            self._chunk_to_concepts[chunk_id] = concepts
            for concept in concepts:
                self._concept_to_chunks.setdefault(concept, set()).add(chunk_id)

            # 同一块内概念共现
            for concept in concepts:
                self._cooccur.setdefault(concept, set()).update(concepts - {concept})

        logger.info(
            f"知识图谱构建完成: {len(self._chunks)}块, "
            f"{len(self._concept_to_chunks)}个概念"
        )

    @staticmethod
    def _match_terms(text: str) -> set[str]:
        """最长匹配优先，返回文本中出现的概念。"""
        matched: set[str] = set()
        for term in _TERMS_BY_LENGTH:
            if term in text:
                matched.add(term)
        return matched

    # ---------- 检索 ----------

    def query_candidates(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """返回与查询概念相关的块（按直接/间接关联度排序）。"""
        if not self._chunks:
            return []

        hit_concepts = self._match_terms(query)
        scores: dict[str, int] = {}

        # 直接关联：概念命中的块
        for concept in hit_concepts:
            for chunk_id in self._concept_to_chunks.get(concept, set()):
                scores[chunk_id] = scores.get(chunk_id, 0) + 2
            # 共现概念关联的块
            for related in self._cooccur.get(concept, set()):
                for chunk_id in self._concept_to_chunks.get(related, set()):
                    scores[chunk_id] = scores.get(chunk_id, 0) + 1

        # 标题命中
        for title in self._titles:
            if title in query or query in title:
                for chunk_id in self._title_to_chunks.get(title, set()):
                    scores[chunk_id] = scores.get(chunk_id, 0) + 2

        if not scores:
            return []

        ranked = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )[:top_k]
        return [self._chunks[chunk_id] for chunk_id, _ in ranked]

    # ---------- 持久化 ----------

    def save(self, path: str | Path) -> None:
        data = {
            "chunks": self._chunks,
            "concept_to_chunks": {
                concept: sorted(ids) for concept, ids in self._concept_to_chunks.items()
            },
            "chunk_to_concepts": {
                chunk_id: sorted(concepts)
                for chunk_id, concepts in self._chunk_to_concepts.items()
            },
            "cooccur": {
                concept: sorted(related)
                for concept, related in self._cooccur.items()
            },
            "titles": self._titles,
            "title_to_chunks": {
                title: sorted(ids) for title, ids in self._title_to_chunks.items()
            },
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, path: str | Path) -> None:
        file = Path(path)
        if not file.exists():
            return
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            self._chunks = data.get("chunks", {})
            self._concept_to_chunks = {
                concept: set(ids)
                for concept, ids in data.get("concept_to_chunks", {}).items()
            }
            self._chunk_to_concepts = {
                chunk_id: set(concepts)
                for chunk_id, concepts in data.get("chunk_to_concepts", {}).items()
            }
            self._cooccur = {
                concept: set(related)
                for concept, related in data.get("cooccur", {}).items()
            }
            self._titles = data.get("titles", [])
            self._title_to_chunks = {
                title: set(ids)
                for title, ids in data.get("title_to_chunks", {}).items()
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"知识图谱文件损坏，已忽略: {e}")
            self._reset()

    def _reset(self) -> None:
        self._chunks = {}
        self._concept_to_chunks = {}
        self._chunk_to_concepts = {}
        self._cooccur = {}
        self._titles = []
        self._title_to_chunks = {}
