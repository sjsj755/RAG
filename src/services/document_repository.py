"""文档注册表持久化：JSON 文件的加载与原子写入。"""

import json
from pathlib import Path

from src.core.models import DocumentInfo
from src.utils.helpers import ensure_dir
from src.utils.logger import logger


class DocumentRepository:
    """将文档元数据（DocumentInfo）持久化到 JSON 文件。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, DocumentInfo]:
        """加载注册表；文件缺失返回空表，损坏时重置并告警。"""
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {
                doc_id: DocumentInfo(**data)
                for doc_id, data in raw.items()
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"文档注册表损坏，已重置: {e}")
            return {}

    def save(self, documents: dict[str, DocumentInfo]) -> None:
        """原子写入注册表（临时文件 + 替换）。"""
        ensure_dir(self._path.parent)
        tmp_path = self._path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    doc_id: doc.model_dump(mode="json")
                    for doc_id, doc in documents.items()
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(self._path)
