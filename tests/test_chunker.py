from src.core.chunker import SemanticChunker, _split_sentences, _token_len


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

    # 多个切分块
    content_chunks = [c["text"] for c in chunks if c["type"] == "text"]
    assert len(content_chunks) > 1

    # 所有块以句子边界（句号）结尾
    for piece in content_chunks:
        assert piece.rstrip().endswith("。")

    # 相邻块存在公共内容（句子级重叠）
    for i in range(len(content_chunks) - 1):
        assert content_chunks[i][-20:] in content_chunks[i + 1]


def test_overlap_ratio_within_tolerance():
    text = "".join(
        f"第{i}个句子主要讨论集合与元素的基本概念。" for i in range(120)
    )
    max_tokens = 200
    chunker = SemanticChunker(max_tokens=max_tokens, overlap_ratio=0.15)
    chunks = chunker.chunk(_page([{"text": text, "type": "text", "block_id": "1"}]))
    content_chunks = [c["text"] for c in chunks if c["type"] == "text"]
    assert len(content_chunks) > 1

    for i in range(len(content_chunks) - 1):
        current = content_chunks[i]
        following = content_chunks[i + 1]
        # 计算公共尾部/头部内容的 token 占比
        overlap_len = 0
        for size in range(min(len(current), len(following)), 0, -1):
            if current[-size:] == following[:size]:
                overlap_len = size
                break
        ratio = _token_len(current[-overlap_len:]) / _token_len(current) if overlap_len else 0.0
        assert 0.10 <= ratio <= 0.30, f"重叠比例异常: {ratio:.3f}"


def test_overlong_sentence_stands_alone():
    # 无标点超长句（如长公式串），不硬切，独立成块
    long_sentence = "这是一个没有标点的超长句子" * 30
    chunker = SemanticChunker(max_tokens=100, overlap_ratio=0.15)
    chunks = chunker.chunk(_page([{"text": long_sentence, "type": "text", "block_id": "1"}]))
    assert len(chunks) == 1
    assert chunks[0]["text"] == long_sentence


def test_formula_not_split():
    formula = "$$ A=\\{0,1,2,3,4,5,6,7,8,9\\} $$"
    text = f"这是一个关于集合的句子。{formula}这也是一个句子。"
    chunker = SemanticChunker(max_tokens=100, overlap_ratio=0.15)
    chunks = chunker.chunk(_page([{"text": text, "type": "text", "block_id": "1"}]))
    combined = "".join(c["text"] for c in chunks if c["type"] == "text")
    assert formula in combined


def test_split_sentences_keeps_decimal():
    sentences = _split_sentences("圆周率约等于3.14。下一个句子开始。")
    assert any("3.14。" in s for s in sentences)


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


def test_build_subchunks_splits_long_parent():
    text = "集合是数学的基础。" * 40
    chunk = {
        "text": text,
        "type": "text",
        "page_num": 3,
        "block_id": "b1",
        "parent_id": "parent_0",
        "chunk_version": 2,
    }
    subs = SemanticChunker().build_subchunks([chunk], max_tokens=100)

    assert len(subs) > 1
    for sub in subs:
        assert sub["parent_id"] == "parent_0"
        assert sub["parent_text"] == text
        assert sub["chunk_version"] == 3
        assert sub["page_num"] == 3
        assert sub["text"].rstrip().endswith("。")
        assert _token_len(sub["text"]) <= 100


def test_build_subchunks_short_parent_single():
    chunk = {
        "text": "短文本",
        "type": "text",
        "page_num": 1,
        "block_id": "b1",
        "parent_id": "parent_0",
    }
    subs = SemanticChunker().build_subchunks([chunk], max_tokens=160)
    assert len(subs) == 1
    assert subs[0]["text"] == "短文本"
    assert subs[0]["parent_text"] == "短文本"


def test_build_subchunks_overlong_sentence_stands_alone():
    long_text = "这是一个没有标点的超长句子" * 30
    chunk = {
        "text": long_text,
        "type": "text",
        "page_num": 1,
        "block_id": "b1",
        "parent_id": "parent_0",
    }
    subs = SemanticChunker().build_subchunks([chunk], max_tokens=100)
    assert len(subs) == 1
    assert subs[0]["text"] == long_text


def test_token_len_fallback():
    assert _token_len("中文测试") >= 1
