"""LLM 答案生成器单元测试。"""

import pytest

from src.core.answer_generator import LLMAnswerGenerator
from src.core.config import Settings


def make_config(**overrides) -> Settings:
    defaults = {
        "qwen_api_key": "k",
        "qwen_base_url": "https://example.com/v1",
        "paddleocr_api_key": "p",
        "deepseek_api_key": "ds-key",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


class FakeClient:
    def __init__(
        self, content: str = "", pieces: list[str] | None = None
    ) -> None:
        self.content = content
        self.pieces = pieces or []
        self.calls: list[dict] = []

    class Completions:
        def __init__(self, client):
            self.client = client

        def create(self, **kwargs):
            self.client.calls.append(kwargs)
            if kwargs.get("stream"):
                return [
                    type(
                        "Chunk",
                        (),
                        {
                            "choices": [
                                type(
                                    "Choice",
                                    (),
                                    {
                                        "delta": type(
                                            "Delta",
                                            (),
                                            {"content": piece},
                                        )()
                                    },
                                )()
                            ]
                        },
                    )()
                    for piece in self.client.pieces
                ]
            message = type("Message", (), {"content": self.client.content})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    @property
    def chat(self):
        return type("Chat", (), {"completions": self.Completions(self)})()


class BoomCompletions(FakeClient.Completions):
    def create(self, **kwargs):
        raise RuntimeError("api down")


class BoomClient(FakeClient):
    @property
    def chat(self):
        return type("Chat", (), {"completions": BoomCompletions(self)})()


def _sources() -> list[dict]:
    return [
        {"index": 1, "page_num": 9, "text": "集合的定义内容"},
        {"index": 2, "page_num": 10, "text": "元素的互异性"},
    ]


def test_build_messages_contains_numbered_sources():
    generator = LLMAnswerGenerator(config=make_config())
    messages = generator._build_messages("什么是集合？", _sources())
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "问题：什么是集合？" in user
    assert "[1] (第9页) 集合的定义内容" in user
    assert "[2] (第10页) 元素的互异性" in user


def test_build_messages_truncates_long_source():
    generator = LLMAnswerGenerator(config=make_config())
    long_text = "长" * 1000
    messages = generator._build_messages(
        "问题", [{"index": 1, "page_num": 1, "text": long_text}]
    )
    assert len(long_text) > 600
    assert "长" * 600 in messages[1]["content"]
    assert "长" * 601 not in messages[1]["content"]


def test_generate_returns_content(monkeypatch):
    generator = LLMAnswerGenerator(config=make_config())
    monkeypatch.setattr(generator, "_client", FakeClient(content="定义答案"))
    assert generator.generate("问题", _sources()) == "定义答案"


def test_generate_without_client_raises():
    generator = LLMAnswerGenerator(
        config=make_config(deepseek_api_key="")
    )
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        generator.generate("问题", _sources())


def test_stream_yields_pieces(monkeypatch):
    generator = LLMAnswerGenerator(config=make_config())
    monkeypatch.setattr(
        generator, "_client", FakeClient(pieces=["定", "义", "答", "案"])
    )
    assert list(generator.stream("问题", _sources())) == ["定", "义", "答", "案"]


def test_generate_propagates_client_error(monkeypatch):
    generator = LLMAnswerGenerator(config=make_config())
    monkeypatch.setattr(generator, "_client", BoomClient())
    with pytest.raises(RuntimeError, match="api down"):
        generator.generate("问题", _sources())
