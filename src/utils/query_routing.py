"""查询类型路由：识别章节/目录类查询，用于改写与融合策略。"""

from __future__ import annotations

import re

_CHAPTER_WORDS = ("章", "节", "目录", "小节", "栏目")
_CHAPTER_PATTERN = re.compile(r"第[一二三四五六七八九十百0-9]+章")
_CONCEPT_WORDS = (
    "是什么",
    "什么是",
    "什么叫",
    "定义",
    "概念",
    "性质",
    "要素",
    "特征",
    "指什么",
    "叫做",
    "含义",
)


def is_chapter_query(query: str) -> bool:
    """查询涉及教材章节/目录/栏目结构时返回 True。"""
    return bool(_CHAPTER_PATTERN.search(query)) or any(
        word in query for word in _CHAPTER_WORDS
    )


def is_concept_query(query: str) -> bool:
    """查询为概念/定义/性质类问题时返回 True。"""
    return any(word in query for word in _CONCEPT_WORDS)
