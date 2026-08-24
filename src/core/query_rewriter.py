"""查询改写器：LLM 生成变体 + embedding 相似度过滤。"""

import json

from openai import OpenAI

from src.core.config import Settings, get_settings
from src.core.protocols import EmbedderProtocol
from src.utils.logger import logger

_SYSTEM_PROMPT = (
    "你是高中数学教材检索系统的查询改写助手。"
    "根据用户的原始问题生成 {count} 个不同的检索查询变体："
    "第一个是同义改写，保持原意；"
    "第二个是关键词扩展版本，补充相关数学术语或拆解为子问题。"
    "只输出 JSON 数组（如 [\"变体1\", \"变体2\"]），不要输出其他内容。"
)


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度。"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


class DeepSeekQueryRewriter:
    """DeepSeek 查询改写 + 相似度过滤。

    - 未配置 API key 或调用失败时返回空列表（调用方回退原始查询）；
    - 生成变体与原始查询的余弦相似度低于阈值时丢弃。
    """

    def __init__(
        self,
        config: Settings | None = None,
        embedder: EmbedderProtocol | None = None,
    ) -> None:
        cfg = config or get_settings()
        self._embedder = embedder
        self._threshold = cfg.query_similarity_threshold
        self._count = cfg.query_rewrite_count
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

    def rewrite(self, query: str) -> list[str]:
        """返回通过相似度过滤的改写查询；失败或未配置时返回空列表。"""
        if self._client is None:
            logger.debug("未配置 DEEPSEEK_API_KEY，跳过查询改写")
            return []
        try:
            candidates = self._generate(query)
        except Exception as e:
            logger.warning(f"查询改写失败，回退原始查询: {e}")
            return []
        return self._filter(query, candidates)

    def _generate(self, query: str) -> list[str]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(count=self._count),
                },
                {"role": "user", "content": query},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        content = response.choices[0].message.content or ""
        return self._parse_json_list(content)

    @staticmethod
    def _parse_json_list(content: str) -> list[str]:
        content = content.strip()
        # 兼容 ```json ... ``` 代码块包裹
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:].lstrip()
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = [
                line.strip().strip('"').strip(",")
                for line in content.splitlines()
                if line.strip()
            ]
        if not isinstance(data, list):
            return []
        return [str(item).strip() for item in data if str(item).strip()]

    def _filter(self, query: str, candidates: list[str]) -> list[str]:
        if not candidates or self._embedder is None:
            return candidates
        try:
            vectors = self._embedder.embed([query, *candidates])
        except Exception as e:
            logger.warning(f"改写查询相似度计算失败，直接使用生成结果: {e}")
            return candidates

        query_vector = vectors[0]
        kept: list[str] = []
        for candidate, vector in zip(candidates, vectors[1:]):
            similarity = _cosine(query_vector, vector)
            if similarity >= self._threshold:
                kept.append(candidate)
            else:
                logger.debug(
                    f"丢弃低相似度改写查询 ({similarity:.3f}): {candidate}"
                )
        return kept
