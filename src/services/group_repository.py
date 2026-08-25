"""分组注册表持久化：JSON 文件的加载与原子写入。"""

import json
from pathlib import Path

from src.core.models import GroupInfo
from src.utils.helpers import ensure_dir
from src.utils.logger import logger


class GroupRepository:
    """将分组元数据（GroupInfo）持久化到 JSON 文件。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, GroupInfo]:
        """加载分组表；文件缺失返回空表，损坏时重置并告警。"""
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {
                group_id: GroupInfo(**data)
                for group_id, data in raw.items()
            }
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"分组注册表损坏，已重置: {e}")
            return {}

    def save(self, groups: dict[str, GroupInfo]) -> None:
        """原子写入分组表（临时文件 + 替换）。"""
        ensure_dir(self._path.parent)
        tmp_path = self._path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    group_id: group.model_dump(mode="json")
                    for group_id, group in groups.items()
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp_path.replace(self._path)
