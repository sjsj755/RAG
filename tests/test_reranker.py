"""LLM 重排器单元测试。"""

from src.core.config import Settings
from src.core.reranker import LLMReranker


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


def _candidates(count: int = 5) -> list[dict]:
    return [{"text": f"块{i}", "page_num": i} for i in range(count)]


def _texts(items: list[dict]) -> list[str]:
    return [item["text"] for item in items]


def test_rerank_orders_by_llm_indices(monkeypatch):
    reranker = LLMReranker(config=make_config())
    monkeypatch.setattr(reranker, "_client", FakeClient("[2, 0, 4]"))
    out = reranker.rerank("什么是集合？", _candidates(), top_n=3)
    assert _texts(out) == ["块2", "块0", "块4"]


def test_rerank_fills_remaining_from_original_order(monkeypatch):
    reranker = LLMReranker(config=make_config())
    monkeypatch.setattr(reranker, "_client", FakeClient("[2]"))
    out = reranker.rerank("问题", _candidates(), top_n=3)
    assert _texts(out) == ["块2", "块0", "块1"]


def test_rerank_empty_result_falls_back(monkeypatch):
    reranker = LLMReranker(config=make_config())
    monkeypatch.setattr(reranker, "_client", FakeClient("[]"))
    out = reranker.rerank("问题", _candidates(), top_n=3)
    assert _texts(out) == ["块0", "块1", "块2"]


def test_rerank_invalid_json_falls_back(monkeypatch):
    reranker = LLMReranker(config=make_config())
    monkeypatch.setattr(reranker, "_client", FakeClient("不是 JSON"))
    out = reranker.rerank("问题", _candidates(), top_n=3)
    assert _texts(out) == ["块0", "块1", "块2"]


def test_rerank_exception_falls_back(monkeypatch):
    reranker = LLMReranker(config=make_config())

    def boom(query, candidates, top_n):
        raise RuntimeError("api down")

    monkeypatch.setattr(reranker, "_select", boom)
    out = reranker.rerank("问题", _candidates(), top_n=3)
    assert _texts(out) == ["块0", "块1", "块2"]


def test_rerank_without_client_returns_original(monkeypatch):
    reranker = LLMReranker(config=make_config(deepseek_api_key=""))
    out = reranker.rerank("问题", _candidates(), top_n=3)
    assert _texts(out) == ["块0", "块1", "块2"]


def test_parse_indices_ignores_out_of_range_and_code_fence():
    assert LLMReranker._parse_indices("[0, 9, -1, 1]", limit=3) == [0, 1]
    assert LLMReranker._parse_indices('```json\n[2,1]\n```', limit=3) == [2, 1]
