"""RAGEngine 离线集成测试：真实 ChromaDB + 假解析器/嵌入器。"""

from src.core.config import settings
from src.core.rag_engine import RAGEngine


class FakeParser:
    def parse(self, pdf_path: str) -> list[dict]:
        return [
            {
                "page_num": 1,
                "blocks": [
                    {
                        "text": "函数定义",
                        "type": "text",
                        "block_id": "b1",
                        "page_num": 1,
                    },
                    {
                        "text": "导数的几何意义",
                        "type": "text",
                        "block_id": "b2",
                        "page_num": 1,
                    },
                ],
            },
            {
                "page_num": 2,
                "blocks": [
                    {
                        "text": "定积分",
                        "type": "display_formula",
                        "block_id": "b3",
                        "page_num": 2,
                    }
                ],
            },
        ]


class FakeEmbedder:
    """用关键词构造确定性向量，便于离线验证相似度排序。"""

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0, 0.0, 0.0, 0.0]
        if "函数" in text:
            vec[0] = 1.0
        if "导数" in text:
            vec[1] = 1.0
        if "积分" in text:
            vec[2] = 1.0
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


def test_build_index_and_search(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))

    engine = RAGEngine(
        collection_name="default",
        parser=FakeParser(),
        embedder=FakeEmbedder(),
    )

    result = engine.build_index("fake.pdf")
    assert result["total_chunks"] == 3
    assert result["pages"] == 2
    assert engine.indexer.count() == 3

    results = engine.search("导数的几何意义", top_k=3)
    assert results, "应返回检索结果"
    assert results[0]["text"] == "导数的几何意义"
    assert results[0]["score"] > 0.9
    assert results[0]["page_num"] == 1
    assert results[0]["block_id"] == "b2"
