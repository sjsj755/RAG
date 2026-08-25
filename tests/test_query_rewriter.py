import pytest

from src.core.config import Settings
from src.core.query_rewriter import (
    _EXAMPLES,
    _SYSTEM_PROMPT,
    DeepSeekQueryRewriter,
    _cosine,
)


def make_config(**overrides) -> Settings:
    defaults = {
        "qwen_api_key": "k",
        "qwen_base_url": "https://example.com/v1",
        "paddleocr_api_key": "p",
        "deepseek_api_key": "ds-key",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


class FakeEmbedder:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.vectors


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content

    class Completions:
        def __init__(self, client):
            self.client = client

        def create(self, **kwargs):
            message = type("Message", (), {"content": self.client.content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    @property
    def chat(self):
        return type("Chat", (), {"completions": self.Completions(self)})()


def test_parse_json_list():
    assert DeepSeekQueryRewriter._parse_json_list('["a", "b"]') == ["a", "b"]
    assert DeepSeekQueryRewriter._parse_json_list(
        '```json\n["a", "b"]\n```'
    ) == ["a", "b"]
    assert DeepSeekQueryRewriter._parse_json_list("a\nb") == ["a", "b"]
    # 非 JSON 时降级为逐行提取
    assert DeepSeekQueryRewriter._parse_json_list("not json") == ["not json"]


def test_system_prompt_contains_care_sections():
    prompt = _SYSTEM_PROMPT.format(count=2)
    for section in ("CONTEXT", "ASK", "RULES", "EXAMPLES"):
        assert section in prompt


def test_parse_cleans_dedupes_and_truncates():
    long_text = "长" * 201
    parsed = DeepSeekQueryRewriter._parse_json_list(
        f'[" 变体 ", "变体", "", "   ", "变体", "{long_text}"]'
    )
    assert parsed == ["变体"]


def test_generate_builds_few_shot_messages(monkeypatch):
    rewriter = DeepSeekQueryRewriter(config=make_config(), embedder=None)
    captured = {}

    class FakeClient:
        def __init__(self):
            self.rewriter = rewriter

        class Completions:
            def __init__(self, client):
                self.client = client

            def create(self, **kwargs):
                captured["messages"] = kwargs["messages"]
                captured["temperature"] = kwargs["temperature"]
                message = type("Message", (), {"content": '["改写一"]'})()
                choice = type("Choice", (), {"message": message})()
                return type("Response", (), {"choices": [choice]})()

        @property
        def chat(self):
            return type("Chat", (), {"completions": self.Completions(self)})()

    monkeypatch.setattr(rewriter, "_client", FakeClient())
    assert rewriter._generate("原始问题") == ["改写一"]

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert len(messages) == 1 + 2 * len(_EXAMPLES) + 1
    assert messages[-1] == {"role": "user", "content": "原始问题"}
    for i, (example_query, example_output) in enumerate(_EXAMPLES):
        assert messages[1 + i * 2] == {
            "role": "user",
            "content": example_query,
        }
        assert messages[2 + i * 2] == {
            "role": "assistant",
            "content": example_output,
        }
    assert captured["temperature"] == 0.2


def test_filter_keeps_above_threshold():
    # 原始查询向量 [1,0,0]，候选 [1,0,0]（相似 1.0）与 [0,1,0]（相似 0）
    embedder = FakeEmbedder([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rewriter = DeepSeekQueryRewriter(
        config=make_config(query_similarity_threshold=0.75), embedder=embedder
    )
    kept = rewriter._filter("原始", ["同义改写", "无关改写"])
    assert kept == ["同义改写"]


def test_rewrite_disabled_without_key():
    rewriter = DeepSeekQueryRewriter(
        config=make_config(deepseek_api_key=""),
        embedder=FakeEmbedder([[1.0], [1.0]]),
    )
    assert rewriter.rewrite("问题") == []


def test_rewrite_failure_falls_back(monkeypatch):
    rewriter = DeepSeekQueryRewriter(
        config=make_config(),
        embedder=FakeEmbedder([[1.0], [1.0]]),
    )

    def boom(query):
        raise RuntimeError("api down")

    monkeypatch.setattr(rewriter, "_generate", boom)
    assert rewriter.rewrite("问题") == []


def test_generate_parses_client_response(monkeypatch):
    rewriter = DeepSeekQueryRewriter(
        config=make_config(),
        embedder=FakeEmbedder([[1.0], [1.0]]),
    )
    monkeypatch.setattr(
        rewriter,
        "_client",
        FakeClient('["改写一", "改写二"]'),
    )
    assert rewriter._generate("原始") == ["改写一", "改写二"]


def test_cosine():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
