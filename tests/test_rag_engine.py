"""RAGEngine 集成测试：真实分块器 + 假解析/嵌入/改写/图谱。"""

from src.core.config import Settings, settings
from src.core.rag_engine import RAGEngine


class FakeParser:
    def parse(self, pdf_path: str) -> list[dict]:
        return [
            {
                "page_num": 1,
                "blocks": [
                    {
                        "text": "1.1 集合的概念",
                        "type": "paragraph_title",
                        "block_id": "0",
                        "page_num": 1,
                    },
                    {
                        "text": "把研究对象统称为元素",
                        "type": "text",
                        "block_id": "1",
                        "page_num": 1,
                    },
                    {
                        "text": "元素具有确定性",
                        "type": "text",
                        "block_id": "2",
                        "page_num": 1,
                    },
                ],
            },
            {
                "page_num": 2,
                "blocks": [
                    {
                        "text": "常用数集",
                        "type": "paragraph_title",
                        "block_id": "3",
                        "page_num": 2,
                    },
                    {
                        "text": "自然数集记作 N",
                        "type": "text",
                        "block_id": "4",
                        "page_num": 2,
                    },
                ],
            },
        ]


class FakeEmbedder:
    """记录被编码的文本序列；关键词构造确定性向量。"""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        vec = [0.0, 0.0, 0.0, 0.0]
        if "集合" in text:
            vec[0] = 1.0
        if "数集" in text:
            vec[1] = 1.0
        if "改写" in text:
            vec[2] = 1.0
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [self._vector(text) for text in texts]


class FakeRewriter:
    def __init__(self, rewritten: list[str] | None = None) -> None:
        self.rewritten = rewritten

    def rewrite(self, query: str) -> list[str]:
        return list(self.rewritten or [])


class FakeKnowledgeGraph:
    def __init__(self, candidates: list[dict] | None = None) -> None:
        self.candidates = candidates or []
        self.built_chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        self.built_chunks = chunks

    def query_candidates(self, query: str, top_k: int = 5) -> list[dict]:
        return list(self.candidates)

    def save(self, path) -> None:
        pass

    def load(self, path) -> None:
        pass


class FakeVectorIndexer:
    """固定分数向量索引器：query 返回按位置递减的 score。"""

    def __init__(self) -> None:
        self.data: list[dict] = []
        self.collection = type(
            "FakeCollection",
            (),
            {
                "get": lambda self, **kwargs: {
                    "documents": [],
                    "metadatas": [],
                }
            },
        )()

    def clear(self) -> None:
        self.data = []

    def drop(self) -> None:
        pass

    def add(self, chunks: list[dict], embeddings: list[list[float]]) -> None:
        self.data.extend(chunks)

    def query(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        return [
            dict(chunk, score=0.5 - i * 0.01)
            for i, chunk in enumerate(self.data[:top_k])
        ]

    def count(self) -> int:
        return len(self.data)


class FakeReranker:
    def __init__(
        self, order: list[int] | None = None, fail: bool = False
    ) -> None:
        self.order = order or []
        self.fail = fail
        self.calls = 0

    def rerank(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("rerank down")
        ordered: list[dict] = []
        seen: set[int] = set()
        for idx in self.order:
            if idx < len(candidates) and idx not in seen:
                ordered.append(candidates[idx])
                seen.add(idx)
        for i, candidate in enumerate(candidates):
            if len(ordered) >= top_n:
                break
            if i not in seen:
                ordered.append(candidate)
                seen.add(i)
        return ordered


def make_engine(
    monkeypatch,
    tmp_path,
    rewriter: FakeRewriter | None = None,
    kg: FakeKnowledgeGraph | None = None,
) -> RAGEngine:
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    return RAGEngine(
        collection_name="default",
        parser=FakeParser(),
        embedder=FakeEmbedder(),
        query_rewriter=rewriter if rewriter else FakeRewriter(),
        knowledge_graph=kg if kg else FakeKnowledgeGraph(),
    )


def test_build_index_chunks_and_builds_graph(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path)

    result = engine.build_index("fake.pdf")

    # 2 个标题块 + 2 个聚合父块
    assert result["total_chunks"] == 4
    assert engine.indexer.count() == 4
    assert len(engine.knowledge_graph.built_chunks) == 4


def test_search_uses_rewritten_query_when_available(monkeypatch, tmp_path):
    engine = make_engine(
        monkeypatch, tmp_path, rewriter=FakeRewriter(["数集改写查询"])
    )
    engine.build_index("fake.pdf")
    engine.embedder.embedded.clear()

    engine.search("原始查询", top_k=3)

    assert engine.embedder.embedded == ["数集改写查询"]


def test_search_falls_back_to_original_query(monkeypatch, tmp_path):
    engine = make_engine(monkeypatch, tmp_path, rewriter=FakeRewriter([]))
    engine.build_index("fake.pdf")
    engine.embedder.embedded.clear()

    engine.search("集合的元素", top_k=3)

    assert engine.embedder.embedded == ["集合的元素"]


def test_search_merges_knowledge_graph_candidates(monkeypatch, tmp_path):
    kg = FakeKnowledgeGraph(
        candidates=[
            {
                "text": "图谱候选块",
                "type": "text",
                "page_num": 9,
                "block_id": "kg_1",
            }
        ]
    )
    engine = make_engine(monkeypatch, tmp_path, kg=kg)
    engine.build_index("fake.pdf")

    results = engine.search("集合", top_k=5)
    texts = [item["text"] for item in results]

    assert "图谱候选块" in texts


def test_chapter_query_keeps_first_variant_and_adds_original(
    monkeypatch, tmp_path
):
    engine = make_engine(
        monkeypatch, tmp_path, rewriter=FakeRewriter(["变体1", "变体2"])
    )
    engine.build_index("fake.pdf")
    engine.embedder.embedded.clear()

    engine.search("第一章包含哪些小节？", top_k=3)

    assert engine.embedder.embedded == ["变体1", "第一章包含哪些小节？"]


def _confidence_engine(tmp_path, threshold: float) -> RAGEngine:
    cfg = Settings(
        _env_file=None,
        qwen_api_key="k",
        qwen_base_url="https://example.com/v1",
        paddleocr_api_key="p",
        answer_confidence_threshold=threshold,
        bm25_enabled=False,
        kg_enabled=False,
        query_rewrite_enabled=False,
        chapter_query_routing=False,
        subchunk_enabled=False,
    )
    return RAGEngine(
        collection_name="t",
        parser=FakeParser(),
        embedder=FakeEmbedder(),
        indexer=FakeVectorIndexer(),
        query_rewriter=FakeRewriter([]),
        knowledge_graph=FakeKnowledgeGraph(),
        config=cfg,
    )


def test_search_with_confidence_refuses_below_threshold(tmp_path):
    engine = _confidence_engine(tmp_path, threshold=0.7)
    engine.build_index("fake.pdf")

    out = engine.search_with_confidence("问题", top_k=5)

    assert out["refused"] is True
    assert out["confidence"] == 0.5  # 单路一致：0.5 × (0.5 + 0.5)
    assert out["results"]


def test_search_with_confidence_keeps_results_when_above_threshold(tmp_path):
    engine = _confidence_engine(tmp_path, threshold=0.4)
    engine.build_index("fake.pdf")

    out = engine.search_with_confidence("问题", top_k=5)

    assert out["refused"] is False
    assert out["results"]


def test_search_with_confidence_threshold_zero_disables_refusal(tmp_path):
    engine = _confidence_engine(tmp_path, threshold=0.0)
    engine.build_index("fake.pdf")

    out = engine.search_with_confidence("问题", top_k=5)

    assert out["refused"] is False


def _rerank_engine(tmp_path, reranker: FakeReranker) -> RAGEngine:
    cfg = Settings(
        _env_file=None,
        qwen_api_key="k",
        qwen_base_url="https://example.com/v1",
        paddleocr_api_key="p",
        bm25_enabled=False,
        kg_enabled=False,
        query_rewrite_enabled=False,
        chapter_query_routing=False,
        subchunk_enabled=False,
        llm_rerank_enabled=True,
    )
    return RAGEngine(
        collection_name="t",
        parser=FakeParser(),
        embedder=FakeEmbedder(),
        indexer=FakeVectorIndexer(),
        query_rewriter=FakeRewriter([]),
        knowledge_graph=FakeKnowledgeGraph(),
        reranker=reranker,
        config=cfg,
    )


def test_concept_query_uses_reranker(tmp_path):
    reranker = FakeReranker(order=[2, 0])
    engine = _rerank_engine(tmp_path, reranker)
    engine.build_index("fake.pdf")

    out = engine.search_with_confidence("什么是指数函数？", top_k=3)

    assert reranker.calls == 1
    texts = [r["text"] for r in out["results"]]
    assert texts[0] == engine.indexer.data[2]["text"]
    assert texts[1] == engine.indexer.data[0]["text"]
    assert out["confidence"] == 0.5  # 重排不改变置信度


def test_non_concept_query_skips_reranker(tmp_path):
    reranker = FakeReranker(order=[2, 0])
    engine = _rerank_engine(tmp_path, reranker)
    engine.build_index("fake.pdf")

    engine.search("用列举法表示小于10的所有自然数组成的集合", top_k=3)

    assert reranker.calls == 0


def test_rerank_failure_falls_back_to_original_order(tmp_path):
    reranker = FakeReranker(order=[2, 0], fail=True)
    engine = _rerank_engine(tmp_path, reranker)
    engine.build_index("fake.pdf")

    out = engine.search_with_confidence("什么是指数函数？", top_k=3)

    assert reranker.calls == 1
    texts = [r["text"] for r in out["results"]]
    assert texts == [engine.indexer.data[i]["text"] for i in range(3)]


def test_restore_parent_text_keeps_fragment():
    items = [
        {
            "text": "命中片段",
            "parent_id": "parent_0",
            "parent_text": "完整父块内容",
        }
    ]
    restored = RAGEngine._restore_parent_text(items)
    assert restored[0]["text"] == "完整父块内容"
    assert restored[0]["fragment"] == "命中片段"


def _add_document_engine(tmp_path) -> RAGEngine:
    cfg = Settings(
        _env_file=None,
        qwen_api_key="k",
        qwen_base_url="https://example.com/v1",
        paddleocr_api_key="p",
        bm25_enabled=False,
        kg_enabled=False,
        query_rewrite_enabled=False,
        chapter_query_routing=False,
        subchunk_enabled=False,
        llm_rerank_enabled=False,
    )
    return RAGEngine(
        collection_name="g1",
        parser=FakeParser(),
        embedder=FakeEmbedder(),
        indexer=FakeVectorIndexer(),
        query_rewriter=FakeRewriter([]),
        knowledge_graph=FakeKnowledgeGraph(),
        config=cfg,
    )


def test_add_document_incremental_with_metadata(tmp_path):
    engine = _add_document_engine(tmp_path)

    first = engine.add_document("fake.pdf", "doc_a", "a.pdf")
    second = engine.add_document("fake.pdf", "doc_b", "b.pdf")

    assert first["total_chunks"] == 4
    assert second["total_chunks"] == 4
    assert engine.indexer.count() == 8  # 增量写入，不清空
    assert all(
        chunk.get("doc_id") == "doc_a" and chunk.get("filename") == "a.pdf"
        for chunk in engine.indexer.data[:4]
    )
    assert all(
        chunk.get("doc_id") == "doc_b" and chunk.get("filename") == "b.pdf"
        for chunk in engine.indexer.data[4:]
    )
