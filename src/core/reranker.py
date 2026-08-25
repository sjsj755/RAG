"""LLM 二阶段重排：DeepSeek 挑选真正回答查询的候选块。"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from src.core.config import Settings, get_settings
from src.utils.logger import logger

_SYSTEM_PROMPT = (
    "你是高中数学教材检索系统的答案重排助手。"
    "根据用户问题，从候选内容块中挑选【能直接回答该概念的定义、性质或含义】的块，"
    "按回答质量从高到低排序。"
    "只输出 JSON 数组，元素为候选块编号（如 [3,1,5]），最多 {top_n} 个；"
    "没有任何块能直接回答时输出 []。不要输出其他内容。"
)


class LLMReranker:
    """DeepSeek 重排器：失败或未配置时回退原顺序。"""

    def __init__(self, config: Settings | None = None) -> None:
        cfg = config or get_settings()
        self._model = cfg.deepseek_model
        self._client = (
            OpenAI(
                api_key=cfg.deepseek_api_key,
                base_url=cfg.deepseek_base_url,
                timeout=30.0,
            )
            if cfg.deepseek_api_key
            else None
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int,
    ) -> list[dict[str, Any]]:
        """返回按 LLM 判断排序的候选（最多 top_n 个），失败回退原顺序。"""
        if not candidates or top_n <= 0:
            return list(candidates[:top_n])
        if self._client is None:
            return list(candidates[:top_n])
        try:
            indices = self._select(query, candidates, top_n)
        except Exception as e:
            logger.warning(f"LLM 重排失败，回退原顺序: {e}")
            return list(candidates[:top_n])
        if not indices:
            return list(candidates[:top_n])

        ordered: list[dict[str, Any]] = []
        seen: set[int] = set()
        for idx in indices:
            if 0 <= idx < len(candidates) and idx not in seen:
                ordered.append(candidates[idx])
                seen.add(idx)
        for i, candidate in enumerate(candidates):
            if len(ordered) >= top_n:
                break
            if i not in seen:
                ordered.append(candidate)
                seen.add(i)
        return ordered

    def _select(self, query: str, candidates: list[dict[str, Any]], top_n: int) -> list[int]:
        lines = [
            f"[{i}] {(c.get('text') or '').replace(chr(10), ' ').strip()[:400]}"
            for i, c in enumerate(candidates)
        ]
        user_content = f"问题：{query}\n\n候选块：\n" + "\n".join(lines)
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(top_n=top_n),
                },
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=128,
        )
        content = response.choices[0].message.content or ""
        return self._parse_indices(content, len(candidates))

    @staticmethod
    def _parse_indices(content: str, limit: int) -> list[int]:
        """解析 JSON 编号数组；非法输入返回空列表。"""
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        indices: list[int] = []
        for item in data:
            try:
                idx = int(item)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < limit:
                indices.append(idx)
        return indices
