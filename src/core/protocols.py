"""核心组件协议（Protocol），用于依赖注入与离线测试替换。"""

from typing import Any, Protocol


class PDFParserProtocol(Protocol):
    """PDF 解析器接口。"""

    def parse(self, pdf_path: str) -> list[dict[str, Any]]: ...


class EmbedderProtocol(Protocol):
    """向量生成器接口。"""

    def embed(self, texts: list[str], max_retries: int = 3) -> list[list[float]]: ...


class VectorIndexerProtocol(Protocol):
    """向量索引器接口。"""

    collection_name: str

    def clear(self) -> None: ...

    def drop(self) -> None: ...

    def add(
        self, chunks: list[dict[str, Any]], embeddings: list[list[float]]
    ) -> None: ...

    def query(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[dict[str, Any]]: ...

    def count(self) -> int: ...
