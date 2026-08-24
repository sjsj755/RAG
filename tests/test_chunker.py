from src.core.chunker import SemanticChunker, _token_len


def _page(blocks: list[dict], page_num: int = 1) -> list[dict]:
    return [{"page_num": page_num, "blocks": blocks}]


def test_groups_by_title_and_merges_content():
    pages = _page(
        [
            {"text": "1.1 集合的概念", "type": "paragraph_title", "block_id": "0"},
            {"text": "定义一", "type": "text", "block_id": "1"},
            {"text": "定义二", "type": "text", "block_id": "2"},
        ]
    )
    chunks = SemanticChunker().chunk(pages)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "1.1 集合的概念"
    assert chunks[0]["type"] == "paragraph_title"
    assert "定义一" in chunks[1]["text"]
    assert "定义二" in chunks[1]["text"]
    assert chunks[1]["parent_id"] == "parent_0"
    assert chunks[1]["chunk_version"] == 2


def test_long_text_split_with_overlap():
    text = ("集合是数学的基础。" * 80)  # 约 800 字符
    chunker = SemanticChunker(max_tokens=100, overlap_ratio=0.15)
    chunks = chunker.chunk(_page([{"text": text, "type": "text", "block_id": "1"}]))

    # 标题块 + 多个滑动窗口块
    content_chunks = [c["text"] for c in chunks if c["type"] == "text"]
    assert len(content_chunks) > 1

    # 相邻窗口应存在公共内容（重叠）
    for i in range(len(content_chunks) - 1):
        assert content_chunks[i][-20:] in content_chunks[i + 1]


def test_no_split_when_short():
    pages = _page([{"text": "短文本", "type": "text", "block_id": "1"}])
    chunks = SemanticChunker(max_tokens=512, overlap_ratio=0.15).chunk(pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "短文本"


def test_no_title_groups_all_together():
    pages = _page(
        [
            {"text": "a", "type": "text", "block_id": "1"},
            {"text": "b", "type": "text", "block_id": "2"},
        ]
    )
    chunks = SemanticChunker().chunk(pages)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "a\nb"


def test_metadata_inherited():
    pages = _page(
        [
            {"text": "标题", "type": "paragraph_title", "block_id": "t1"},
            {"text": "内容", "type": "text", "block_id": "b1"},
        ],
        page_num=7,
    )
    chunks = SemanticChunker().chunk(pages)
    content = chunks[1]
    assert content["page_num"] == 7
    assert content["block_id"] == "b1"


def test_token_len_fallback():
    assert _token_len("中文测试") >= 1
