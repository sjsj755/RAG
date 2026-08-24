"""语义分块器。

策略：以章节/标题为结构边界，同节内相邻内容块聚合成父块；
父块超过 token 上限时，按句子边界贪心组装，相邻块带指定比例的重叠。
"""

import re
from typing import Any

import tiktoken

_TITLE_TYPES = {"doc_title", "paragraph_title"}

_ENCODER: tiktoken.Encoding | None = None

_FORMULA_PATTERN = re.compile(r"\$\$.*?\$\$|\$.*?\$", re.DOTALL)
_SENTENCE_PIECE_PATTERN = re.compile(r"[^。！？；\n.!?;]+[。！？；\n.!?;]?")


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


def _split_plain(part: str) -> list[str]:
    """按句子边界切分普通文本；小数（如 3.14）不会被切开。"""
    pieces = _SENTENCE_PIECE_PATTERN.findall(part)
    merged: list[str] = []
    for piece in pieces:
        if (
            merged
            and piece[:1].isdigit()
            and merged[-1].rstrip().endswith(".")
        ):
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def _split_sentences(text: str) -> list[str]:
    """切分句子：公式段（$...$/$$...$$）视为不可分割整体。"""
    segments: list[str] = []
    last = 0
    for match in _FORMULA_PATTERN.finditer(text):
        if match.start() > last:
            segments.extend(_split_plain(text[last : match.start()]))
        segments.append(match.group())
        last = match.end()
    if last < len(text):
        segments.extend(_split_plain(text[last:]))
    return [segment for segment in segments if segment.strip()] or [text]


def _assemble_sentences(
    sentences: list[str], max_tokens: int, overlap_ratio: float
) -> list[str]:
    """贪心组装句子成块；下一块从上一块尾部按比例重叠的句子边界开始。"""
    if not sentences:
        return []
    if sum(_token_len(sentence) for sentence in sentences) <= max_tokens:
        return ["".join(sentences)]

    token_counts = [_token_len(sentence) for sentence in sentences]
    overlap = int(max_tokens * overlap_ratio)
    n = len(sentences)
    chunks: list[str] = []
    i = 0

    while i < n:
        # 贪心组装当前块：至少包含一句，尽量多装直到超出上限
        current_tokens = 0
        j = i
        while j < n:
            next_tokens = current_tokens + token_counts[j]
            if next_tokens > max_tokens and current_tokens > 0:
                break
            current_tokens = next_tokens
            j += 1

        # 单句超长：独立成块，不硬切（避免切断公式/长串）
        if j == i:
            j = i + 1
            current_tokens = token_counts[i]

        chunks.append("".join(sentences[i:j]))
        if j >= n:
            break

        # 下一块起点：从当前块尾部向前回溯，使重叠部分接近 overlap tokens
        k = j
        accumulated = 0
        while k > i and accumulated < overlap:
            k -= 1
            accumulated += token_counts[k]
        i = k if k > i else j

    return chunks


def _split_long_text(text: str, max_tokens: int, overlap_ratio: float) -> list[str]:
    """句子级切分 + 尾部重叠；tiktoken 不可用时按字符数估算。"""
    if _token_len(text) <= max_tokens:
        return [text]
    return _assemble_sentences(
        _split_sentences(text), max_tokens, overlap_ratio
    )


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
