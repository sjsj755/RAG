"""数学公式文本规范化：统一 LaTeX 写法与空白，索引侧与查询侧对称使用。"""

from __future__ import annotations

import re

_EXP_BRACES = re.compile(r"\^\{([^}]+)\}")
_EXP_PLAIN = re.compile(r"\^([0-9A-Za-z]+)")
_LOG_UNDERSCORE = re.compile(r"\\log_\{([^}]+)\}")


def normalize_math_text(text: str) -> str:
    """规范化公式写法：折叠 ^{}、统一属于符号、折叠 \\log_{a}、归一空白。"""
    norm = text
    norm = norm.replace("\\notin", "∉").replace("\\in", "∈")
    norm = norm.replace("\\not\\in", "∉")
    norm = _EXP_BRACES.sub(r"^\1", norm)
    norm = _EXP_PLAIN.sub(r"^\1", norm)
    norm = _LOG_UNDERSCORE.sub(r"log_\1", norm)
    return " ".join(norm.split())
