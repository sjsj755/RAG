from src.core.knowledge_graph import RuleKnowledgeGraph


def make_chunks():
    return [
        {
            "text": "集合是数学的基础，元素具有确定性。",
            "type": "text",
            "page_num": 9,
            "block_id": "1",
        },
        {
            "text": "函数是刻画变量变化的模型，定义域和值域是核心。",
            "type": "text",
            "page_num": 5,
            "block_id": "2",
        },
        {
            "text": "1.1 集合的概念",
            "type": "paragraph_title",
            "page_num": 9,
            "block_id": "0",
        },
    ]


def test_build_extracts_concepts():
    graph = RuleKnowledgeGraph()
    graph.build(make_chunks())
    assert "集合" in graph._concept_to_chunks
    assert "函数" in graph._concept_to_chunks
    assert "定义域" in graph._concept_to_chunks


def test_query_candidates_returns_related_chunks():
    graph = RuleKnowledgeGraph()
    graph.build(make_chunks())

    candidates = graph.query_candidates("什么是函数？", top_k=3)

    assert len(candidates) > 0
    assert any("函数" in c["text"] for c in candidates)


def test_cooccurrence_expands_recall():
    graph = RuleKnowledgeGraph()
    graph.build(make_chunks())

    # 查询"定义域"应能通过共现召回包含"函数"的块
    candidates = graph.query_candidates("定义域", top_k=3)
    assert any("函数" in c["text"] for c in candidates)


def test_title_match():
    graph = RuleKnowledgeGraph()
    graph.build(make_chunks())
    candidates = graph.query_candidates("集合的概念", top_k=3)
    assert any(c["type"] == "paragraph_title" for c in candidates)


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "kg.json"
    graph = RuleKnowledgeGraph()
    graph.build(make_chunks())
    graph.save(path)

    loaded = RuleKnowledgeGraph()
    loaded.load(path)
    assert loaded.query_candidates("函数", top_k=3)
    assert "集合" in loaded._concept_to_chunks


def test_empty_graph_returns_empty():
    graph = RuleKnowledgeGraph()
    assert graph.query_candidates("函数", top_k=3) == []


def test_load_many_merges_docs_with_doc_id(tmp_path):
    path_a = tmp_path / "doc_a.kg.json"
    path_b = tmp_path / "doc_b.kg.json"
    graph_a = RuleKnowledgeGraph()
    graph_a.build(make_chunks())
    graph_a.save(path_a)
    graph_b = RuleKnowledgeGraph()
    graph_b.build(
        [
            {
                "text": "三角函数的周期性很重要。",
                "type": "text",
                "page_num": 20,
                "block_id": "1",
            }
        ]
    )
    graph_b.save(path_b)

    merged = RuleKnowledgeGraph()
    merged.load_many(
        [("doc_a", path_a), ("doc_b", path_b)],
        doc_names={"doc_a": "a.pdf", "doc_b": "b.pdf"},
    )

    candidates = merged.query_candidates("函数", top_k=10)
    assert candidates
    assert any(c.get("doc_id") == "doc_a" for c in candidates)
    assert any(c.get("doc_id") == "doc_b" for c in candidates)
    assert any(c.get("filename") == "a.pdf" for c in candidates)


def test_load_many_skips_missing_files(tmp_path):
    merged = RuleKnowledgeGraph()
    merged.load_many([("doc_x", tmp_path / "missing.json")])
    assert merged.query_candidates("函数", top_k=3) == []
