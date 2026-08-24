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


class ChunkerProtocol(Protocol):
    """语义分块器接口：将解析后的页面块聚合/切分为检索单元。"""

    def chunk(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


class QueryRewriterProtocol(Protocol):
    """查询改写器接口：返回通过相似度过滤的改写查询（可为空）。"""

    def rewrite(self, query: str) -> list[str]: ...


class KnowledgeGraphProtocol(Protocol):
    """轻量知识图谱接口：构建与检索候选。"""

    def build(self, chunks: list[dict[str, Any]]) -> None: ...

    def query_candidates(
        self, query: str, top_k: int = 5
    ) -> list[dict[str, Any]]: ...

    def save(self, path: str | Any) -> None: ...

    def load(self, path: str | Any) -> None: ...
