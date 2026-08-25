"""LLM 答案生成：基于检索片段生成带引用编号的回答（支持流式）。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from openai import OpenAI

from src.core.config import Settings, get_settings

_SYSTEM_PROMPT = (
    "你是高中数学教材（人教A版必修一）检索系统的答疑助手。\n"
    "要求：\n"
    "1. 严格基于提供的教材片段回答，只使用片段中的内容，不得编造；\n"
    "2. 公式与符号用 LaTeX 书写（如 $x^{2}=x$）；\n"
    "3. 回答开头直接给出结论，再补充解释；\n"
    "4. 引用来源时使用 [n] 标注，n 对应片段编号；\n"
    "5. 如果片段不足以回答问题，明确说明“教材中未找到相关内容”，不要猜测。"
)


class LLMAnswerGenerator:
    """DeepSeek 答案生成器：generate 返回完整文本，stream 逐段产出。"""

    def __init__(self, config: Settings | None = None) -> None:
        cfg = config or get_settings()
        self._model = cfg.deepseek_model
        self._temperature = cfg.answer_temperature
        self._max_tokens = cfg.answer_max_tokens
        self._source_max_chars = 600
        self._client = (
            OpenAI(
                api_key=cfg.deepseek_api_key,
                base_url=cfg.deepseek_base_url,
                timeout=60.0,
            )
            if cfg.deepseek_api_key
            else None
        )

    def generate(self, query: str, sources: list[dict[str, Any]]) -> str:
        """非流式生成完整答案；未配置或调用失败时抛出异常。"""
        self._require_client()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(query, sources),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""

    def stream(
        self, query: str, sources: list[dict[str, Any]]
    ) -> Iterator[str]:
        """流式生成答案片段；未配置或调用失败时抛出异常。"""
        self._require_client()
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(query, sources),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def _require_client(self) -> None:
        if self._client is None:
            raise RuntimeError("未配置 DEEPSEEK_API_KEY，无法生成答案")

    def _build_messages(
        self, query: str, sources: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        blocks = []
        for source in sources:
            text = (source.get("text") or "").replace("\n", " ").strip()
            label = source.get("filename") or source.get("doc_id") or ""
            location = (
                f"{label} 第{source.get('page_num')}页"
                if label
                else f"第{source.get('page_num')}页"
            )
            blocks.append(
                f"[{source.get('index')}] "
                f"({location}) {text[:self._source_max_chars]}"
            )
        user_content = f"问题：{query}\n\n教材片段：\n" + "\n".join(blocks)
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
