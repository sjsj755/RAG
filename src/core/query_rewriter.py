"""查询改写器：CARE 提示词框架 + LLM 生成变体 + embedding 相似度过滤。

CARE（Context / Ask / Rules / Examples）是提示词工程框架（NN/g, 2024）：
- CONTEXT：角色与场景；
- ASK：明确任务与输出格式；
- RULES：约束规则；
- EXAMPLES：few-shot 样例，稳定输出质量。
"""

import json
from typing import Any

from openai import OpenAI

from src.core.config import Settings, get_settings
from src.core.protocols import EmbedderProtocol
from src.utils.logger import logger

_SYSTEM_PROMPT = (
    "CONTEXT：你是高中数学教材（人教A版必修一）检索系统的查询改写助手。"
    "用户问题面向教材内容检索，覆盖集合与常用逻辑用语、函数、"
    "指数函数与对数函数、三角函数等章节。\n\n"
    "ASK：根据用户的原始问题生成 {count} 个不同的检索查询变体："
    "第 1 个是同义改写，保持原意；"
    "第 2 个是关键词扩展版本，补充相关数学术语或拆解为子问题。"
    "只输出 JSON 数组（如 [\"变体1\", \"变体2\"]），不要输出其他内容。\n\n"
    "RULES：\n"
    "1. 每个变体必须保持原始问题的核心语义，不得增删意图；\n"
    "2. 不得引入教材范围外的知识或臆造数学术语；\n"
    "3. 每个变体长度控制在 5-80 字；\n"
    "4. 变体之间在用词和句式上必须有明显差异，不得重复；\n"
    "5. 只输出 JSON 数组，禁止解释、编号或 Markdown 代码块包裹；\n"
    "6. 反例：对\"什么是列举法？\"输出 [\"列举法的历史发展\"] 是错误示例"
    "（偏离原意），正确输出见下方 EXAMPLES。\n\n"
    "EXAMPLES：下方多轮对话给出示例输入与期望输出，"
    "请严格遵循其格式、风格与长度。"
)

_EXAMPLES: list[tuple[str, str]] = [
    (
        "什么是集合？集合中的元素具有哪些基本性质？",
        '["集合与元素的定义是什么？元素有哪些性质？", '
        '"集合概念 元素 确定性 互异性"]',
    ),
    (
        "如何用符号表示一个元素属于某个集合？",
        '["元素属于集合的符号表示方法是什么？", "a∈A 属于符号 集合的表示"]',
    ),
    (
        "人教A版必修一包含哪些章节？",
        '["必修一教材目录包含哪些章节？", '
        '"必修一 章节 集合与常用逻辑用语 函数 指数函数 对数函数 三角函数"]',
    ),
    (
        "什么是列举法？",
        '["列举法如何表示集合？", "列举法 集合表示 花括号 一一列举"]',
    ),
]


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
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT.format(count=self._count),
            }
        ]
        for example_query, example_output in _EXAMPLES:
            messages.append({"role": "user", "content": example_query})
            messages.append({"role": "assistant", "content": example_output})
        messages.append({"role": "user", "content": query})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
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
        return DeepSeekQueryRewriter._clean_candidates(data)

    @staticmethod
    def _clean_candidates(items: list[Any]) -> list[str]:
        """统一清洗：去首尾空白、去空、按出现顺序去重、丢弃超长变体。"""
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in items:
            text = str(item).strip()
            if not text or text in seen or len(text) > 200:
                continue
            seen.add(text)
            cleaned.append(text)
        return cleaned

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
