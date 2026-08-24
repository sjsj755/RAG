"""PDF 文档解析器：基于 PaddleOCR-VL API 提取结构化内容块。"""

import re
from typing import Any
from uuid import uuid4

from paddleocr import PaddleOCRClient

from src.core.config import Settings, get_settings
from src.utils.helpers import to_dict
from src.utils.logger import logger

# 广告推广与版权出版噪声内容特征（命中即过滤，对任意教材通用）
_NOISE_PATTERNS = (
    re.compile(
        r"公众号|微信号|扫码关注|学习方法指导|全套.{0,8}(PDF|电子课本)|"
        r"分享给大家|回复.{0,8}电子课本"
    ),
    re.compile(
        r"ISBN|书号|定价|版权所有|未经许可|审图号|意见反馈平台|责任编辑|"
        r"美术编辑|主编|副主编|编写人员|重印|开本|印张|印刷"
    ),
)


def _is_noise_content(content: str) -> bool:
    """判断内容是否为广告推广或版权出版噪声。"""
    return any(pattern.search(content) for pattern in _NOISE_PATTERNS)


class PDFParser:
    def __init__(self, config: Settings | None = None) -> None:
        cfg = config or get_settings()
        self.client = PaddleOCRClient(
            token=cfg.paddleocr_api_key,
            base_url=cfg.paddleocr_base_url,
        )
        self.model = cfg.paddleocr_model

    def parse(self, pdf_path: str) -> list[dict[str, Any]]:
        """解析 PDF，返回页面列表（每页包含 text/type/block_id/page_num 块）。"""
        logger.info(f"解析PDF: {pdf_path}")
        try:
            result = self.client.parse_document(
                file_path=pdf_path,
                model=self.model,
                batch_id=uuid4().hex,
            )
            result_dict = to_dict(result)
        except Exception as e:
            logger.error(f"PDF解析失败: {e}")
            raise RuntimeError(f"PDF解析失败: {e}") from e

        pages = result_dict.get("pages", [])
        parsed_pages = []

        # API 返回的页面列表不携带页码字段，页码即列表下标（从 1 开始）
        for page_index, page in enumerate(pages, start=1):
            page_num = page.get("page_num") or page_index
            pruned = page.get("pruned_result")
            blocks = []

            if pruned:
                parsing_list = pruned.get("parsing_res_list", [])
                for block in parsing_list:
                    label = block.get("block_label", "")
                    content = block.get("block_content", "").strip()
                    # 忽略页眉页脚、孤立数字、脚注等无关内容
                    # （与 PaddleOCR 官方 markdown 的 markdown_ignore_labels 保持一致）
                    if content and label not in {
                        "header",
                        "footer",
                        "header_image",
                        "footer_image",
                        "number",
                        "footnote",
                        "aside_text",
                    } and not _is_noise_content(content):
                        blocks.append(
                            {
                                "text": content,
                                "type": label,
                                "block_id": str(block.get("block_id", "")),
                                "page_num": page_num,
                            }
                        )
            else:
                # 降级方案：使用完整 markdown
                markdown = page.get("markdown_text", "").strip()
                if markdown and not _is_noise_content(markdown):
                    blocks.append(
                        {
                            "text": markdown,
                            "type": "markdown",
                            "block_id": f"page_{page_num}",
                            "page_num": page_num,
                        }
                    )

            parsed_pages.append({"page_num": page_num, "blocks": blocks})

        total_blocks = sum(len(p["blocks"]) for p in parsed_pages)
        logger.info(f"解析完成: {len(parsed_pages)}页, {total_blocks}个内容块")
        return parsed_pages
