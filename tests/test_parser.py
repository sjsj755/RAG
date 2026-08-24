from src.core.parser import PDFParser, _is_noise_content


def test_noise_content_detection():
    assert _is_noise_content("请关注公众号“高中生学习方法”领取资料")
    assert _is_noise_content("ISBN 978-7-107-12345-6 定价 15.00 元")
    assert _is_noise_content("版权所有·未经许可不得复制使用本产品")
    assert not _is_noise_content("集合是刻画一类事物的语言和工具")
    assert not _is_noise_content("函数是描述客观世界变化规律的重要数学模型")


def test_parse_filters_ad_and_copyright_blocks(monkeypatch):
    fake_result = {
        "pages": [
            {
                "page_num": 1,
                "pruned_result": {
                    "parsing_res_list": [
                        {
                            "block_label": "text",
                            "block_content": "请关注公众号领取全套电子课本",
                            "block_id": 0,
                        },
                        {
                            "block_label": "algorithm",
                            "block_content": "ISBN 978-7-107-0000 定价 元 版权所有",
                            "block_id": 1,
                        },
                        {
                            "block_label": "text",
                            "block_content": "集合是刻画一类事物的语言和工具",
                            "block_id": 2,
                        },
                    ]
                },
            }
        ]
    }

    parser = PDFParser()
    monkeypatch.setattr(
        parser.client, "parse_document", lambda **kwargs: fake_result
    )

    pages = parser.parse("fake.pdf")
    blocks = pages[0]["blocks"]
    assert len(blocks) == 1
    assert "集合是刻画一类事物的语言和工具" in blocks[0]["text"]


def test_parse_filters_noisy_markdown_fallback(monkeypatch):
    fake_result = {
        "pages": [
            {
                "page_num": 1,
                "pruned_result": None,
                "markdown_text": "欢迎关注公众号获取全套电子课本",
            }
        ]
    }

    parser = PDFParser()
    monkeypatch.setattr(
        parser.client, "parse_document", lambda **kwargs: fake_result
    )

    pages = parser.parse("fake.pdf")
    assert pages[0]["blocks"] == []
