"""PDF 文件存储：负责上传文件的原子写入、路径解析与删除。"""

from collections.abc import Awaitable, Callable
from pathlib import Path

import aiofiles


class FileTooLargeError(Exception):
    """上传文件超过大小限制。"""


class FileStorage:
    """基于本地目录的 PDF 文件存储。"""

    def __init__(self, upload_dir: str | Path) -> None:
        self._dir = Path(upload_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._dir

    def path_for(self, doc_id: str) -> Path:
        """返回文档对应的 PDF 路径。"""
        return self._dir / f"{doc_id}.pdf"

    def exists(self, doc_id: str) -> bool:
        return self.path_for(doc_id).exists()

    def delete(self, doc_id: str) -> None:
        """删除文档文件（不存在时静默）。"""
        self.path_for(doc_id).unlink(missing_ok=True)

    async def save(
        self,
        doc_id: str,
        read_chunk: Callable[[], Awaitable[bytes]],
        max_bytes: int,
    ) -> int:
        """流式写入文件，超限抛 FileTooLargeError，返回实际字节数。

        read_chunk 为异步可调用对象，返回本次读取的字节（EOF 时返回空字节）。
        写入采用临时文件 + 原子替换，失败时清理临时文件。
        """
        final_path = self.path_for(doc_id)
        tmp_path = final_path.with_suffix(".pdf.tmp")
        written = 0
        try:
            async with aiofiles.open(tmp_path, "wb") as f:
                while chunk := await read_chunk():
                    written += len(chunk)
                    if written > max_bytes:
                        raise FileTooLargeError(
                            f"文件大小超过限制 ({max_bytes // 1024 // 1024}MB)"
                        )
                    await f.write(chunk)
            tmp_path.replace(final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        return written
