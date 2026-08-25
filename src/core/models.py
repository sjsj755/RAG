"""API 数据模型（Pydantic schemas）。"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkResponse(BaseModel):
    text: str
    type: str = "text"
    page_num: int = 0
    score: float | None = None
    # PaddleOCR 的 block_id 可能是整数或字符串，统一兼容
    block_id: str | int | None = None
    # 命中子块原文（未子块化时为 None），text 始终为完整父块
    fragment: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    knowledge_base_id: str = "default"


class SearchResponse(BaseModel):
    query: str
    results: list[ChunkResponse]
    total: int
    confidence: float | None = None
    refused: bool = False


class CitationSource(BaseModel):
    index: int
    page_num: int = 0
    text: str
    score: float | None = None


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class AnswerResponse(BaseModel):
    query: str
    answer: str | None = None
    refused: bool = False
    refusal_reason: str | None = None
    confidence: float | None = None
    sources: list[CitationSource] = []


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_size: int
    status: DocumentStatus
    total_chunks: int = 0
    created_at: datetime
    updated_at: datetime
    error: str | None = None


class IndexRequest(BaseModel):
    document_id: str


class IndexResponse(BaseModel):
    document_id: str
    status: str
    total_chunks: int
    message: str


class StatsResponse(BaseModel):
    collection: str
    total_vectors: int
    is_indexed: bool


class DeleteResponse(BaseModel):
    status: str
    document_id: str


class BatchUploadItem(BaseModel):
    """批量上传中的单个文件结果。"""

    filename: str
    status: str  # uploaded | rejected
    doc_id: str | None = None
    file_size: int = 0
    error: str | None = None


class BatchUploadResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[BatchUploadItem]


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
