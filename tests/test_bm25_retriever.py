"""BM25 稀疏检索路单元测试。"""

from src.core.bm25_retriever import BM25Retriever, tokenize


def _chunks() -> list[dict]:
    return [
        {"text": "集合的定义是确定性。", "page_num": 9, "block_id": "1"},
        {"text": "自然数集记作 N。", "page_num": 9, "block_id": "2"},
        {"text": "指数函数与对数函数是第四章。", "page_num": 6, "block_id": "3"},
    ]


def test_tokenize_mixes_terms_bigram_and_ascii():
    tokens = tokenize("正整数集N与x^{2}=x")
    assert "正整数" in tokens
    assert "N" in tokens
    assert "x^{2}=x" in tokens
    assert "函数" in tokenize("函数")


def test_query_returns_exact_match_first():
    retriever = BM25Retriever()
    retriever.build(_chunks())
    results = retriever.query("自然数集", top_k=3)
    assert results
    assert results[0]["text"] == "自然数集记作 N。"


def test_query_with_unknown_terms_returns_empty():
    retriever = BM25Retriever()
    retriever.build(_chunks())
    assert retriever.query("量子纠缠", top_k=3) == []


def test_empty_corpus_safe():
    retriever = BM25Retriever()
    retriever.build([])
    assert retriever.query("集合", top_k=3) == []
    assert retriever.is_built is False
