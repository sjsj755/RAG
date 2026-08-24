"""语义分块器。

策略：以章节/标题为结构边界，同节内相邻内容块聚合成父块；
父块超过 token 上限时，按滑动窗口切分并带指定比例的重叠。
"""

from typing import Any

import tiktoken

_TITLE_TYPES = {"doc_title", "paragraph_title"}

_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("o200k_base")
    return _ENCODER


def _token_len(text: str) -> int:
    try:
        return len(_get_encoder().encode(text))
    except Exception:
        # tiktoken 不可用时退化为字符数估算
        return len(text)


def _split_long_text(text: str, max_tokens: int, overlap_ratio: float) -> list[str]:
    """按 token 滑动窗口切分，窗口间重叠 overlap_ratio。"""
    if _token_len(text) <= max_tokens:
        return [text]

    overlap = int(max_tokens * overlap_ratio)
    step = max(max_tokens - overlap, 1)

    try:
        tokens = _get_encoder().encode(text)
    except Exception:
        # 字符级回退：按字符窗口切分
        return [
            text[i : i + max_tokens]
            for i in range(0, len(text), max(step, 1))
        ]

    windows = []
    start = 0
    total = len(tokens)
    while start < total:
        end = min(start + max_tokens, total)
        windows.append(_get_encoder().decode(tokens[start:end]))
        if end >= total:
            break
        start = end - overlap
        if start <= 0:
            break
    return windows


class SemanticChunker:
    """结构语义分块：标题分节 → 聚合 → 超长块滑动窗口重叠切分。"""

    def __init__(self, max_tokens: int = 512, overlap_ratio: float = 0.15) -> None:
        self.max_tokens = max_tokens
        self.overlap_ratio = overlap_ratio

    def chunk(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """将解析页面块转换为检索 chunk。"""
        raw_blocks = self._flatten(pages)
        parents = self._group_by_title(raw_blocks)

        chunks: list[dict[str, Any]] = []
        for parent_index, parent in enumerate(parents):
            parent_id = f"parent_{parent_index}"
            title = parent.get("title", "")

            # 标题本身保留为独立 chunk，便于按标题检索
            if title and parent.get("title_meta"):
                title_chunk = dict(parent["title_meta"])
                title_chunk.update(
                    {
                        "text": title,
                        "page_num": parent["title_meta"].get("page_num", 0),
                        "parent_id": parent_id,
                        "chunk_version": 2,
                    }
                )
                chunks.append(title_chunk)

            if not parent["blocks"]:
                continue

            text = "\n".join(block["text"] for block in parent["blocks"])
            if title:
                text = f"{title}\n{text}"

            if _token_len(text) <= self.max_tokens:
                pieces = [text]
            else:
                pieces = _split_long_text(
                    text, self.max_tokens, self.overlap_ratio
                )

            first = parent["blocks"][0]
            page_nums = {
                block.get("page_num", 0) for block in parent["blocks"]
            }
            for piece in pieces:
                chunks.append(
                    {
                        "text": piece,
                        "type": first.get("type", "text"),
                        "page_num": min(page_nums) if page_nums else 0,
                        "block_id": str(first.get("block_id", "")),
                        "parent_id": parent_id,
                        "chunk_version": 2,
                    }
                )
        return chunks

    @staticmethod
    def _flatten(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for page in pages:
            page_num = page.get("page_num", 0)
            for block in page.get("blocks", []):
                item = dict(block)
                item.setdefault("page_num", page_num)
                blocks.append(item)
        return blocks

    @staticmethod
    def _group_by_title(
        blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """以标题块为边界，将连续内容块聚合为父块。"""
        parents: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for block in blocks:
            if block.get("type") in _TITLE_TYPES:
                if current is not None:
                    parents.append(current)
                current = {
                    "title": block.get("text", "").strip(),
                    "title_meta": block,
                    "blocks": [],
                }
            else:
                if current is None:
                    current = {"title": "", "title_meta": None, "blocks": []}
                current["blocks"].append(block)

        if current is not None:
            parents.append(current)
        return parents
