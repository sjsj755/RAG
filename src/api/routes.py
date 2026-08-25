"""RAG API 路由。"""

import json

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_app_settings, get_file_storage, get_knowledge_base
from src.core.config import Settings
from src.core.models import (
    AnswerRequest,
    AnswerResponse,
    BatchUploadItem,
    BatchUploadResponse,
    ChunkResponse,
    DeleteResponse,
    DocumentInfo,
    DocumentStatus,
    GroupCreateRequest,
    GroupInfo,
    IndexResponse,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)
from src.services.file_storage import FileStorage, FileTooLargeError
from src.services.knowledge_base import KnowledgeBaseService
from src.utils.logger import logger

router = APIRouter(prefix="/api/v1", tags=["RAG"])

_PDF_MAGIC = b"%PDF-"


def _require_completed_doc(
    kb: KnowledgeBaseService, doc_id: str
) -> DocumentInfo:
    """校验文档存在且已完成索引，返回文档信息。"""
    doc_info = kb.get_document_info(doc_id)
    if not doc_info:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"文档不存在: {doc_id}")
    if doc_info.status != DocumentStatus.COMPLETED:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"文档尚未完成索引，当前状态: {doc_info.status.value}",
        )
    return doc_info


async def _validate_pdf(file: UploadFile, max_upload_size_mb: int) -> int:
    """校验文件类型、魔数、空文件与大小，返回允许的最大字节数。"""
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "仅支持 PDF 文件")

    max_bytes = max_upload_size_mb * 1024 * 1024

    # 优先使用 Content-Length 快速拒绝超大文件
    content_length = file.headers.get("content-length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            declared_size = 0
        if declared_size == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "文件为空")
        if declared_size > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"文件大小超过限制 ({max_upload_size_mb}MB)",
            )

    # 魔数校验：读取前 5 字节后复位游标，不影响后续流式写入
    head = await file.read(len(_PDF_MAGIC))
    await file.seek(0)
    if not head:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "文件为空")
    if head != _PDF_MAGIC:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "不是有效的 PDF 文件"
        )
    return max_bytes


async def _save_upload(
    file: UploadFile,
    kb: KnowledgeBaseService,
    storage: FileStorage,
    max_bytes: int,
) -> DocumentInfo:
    """创建记录、流式保存文件，失败时回滚记录。"""
    doc_id = kb.create_document(file.filename or "unnamed.pdf", 0)
    try:
        written = await storage.save(doc_id, file.read, max_bytes)
        if written == 0:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "文件为空")
    except FileTooLargeError:
        kb.delete_document(doc_id)
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"文件大小超过限制 ({max_bytes // 1024 // 1024}MB)",
        )
    except HTTPException:
        kb.delete_document(doc_id)
        raise
    except Exception:
        kb.delete_document(doc_id)
        raise

    kb.update_document_file_size(doc_id, written)
    doc = kb.get_document_info(doc_id)
    assert doc is not None
    return doc


@router.post(
    "/upload",
    response_model=DocumentInfo,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf(
    file: UploadFile = File(...),
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
    storage: FileStorage = Depends(get_file_storage),
    settings: Settings = Depends(get_app_settings),
):
    """上传 PDF 文件并创建文档记录。"""
    max_bytes = await _validate_pdf(file, settings.max_upload_size_mb)
    return await _save_upload(file, kb, storage, max_bytes)


@router.post(
    "/upload/batch",
    response_model=BatchUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_pdf_batch(
    files: list[UploadFile] = File(...),
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
    storage: FileStorage = Depends(get_file_storage),
    settings: Settings = Depends(get_app_settings),
):
    """批量上传 PDF：串行处理，单文件失败不影响其他文件。"""
    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"一次最多上传 {settings.max_batch_files} 个文件",
        )

    results: list[BatchUploadItem] = []
    for file in files:
        filename = file.filename or "unnamed.pdf"
        try:
            max_bytes = await _validate_pdf(file, settings.max_upload_size_mb)
            doc = await _save_upload(file, kb, storage, max_bytes)
            results.append(
                BatchUploadItem(
                    filename=filename,
                    status="uploaded",
                    doc_id=doc.id,
                    file_size=doc.file_size,
                )
            )
        except HTTPException as e:
            detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            results.append(
                BatchUploadItem(
                    filename=filename, status="rejected", error=detail
                )
            )
        except Exception as e:
            logger.exception(f"批量上传文件失败: {filename}: {e}")
            results.append(
                BatchUploadItem(
                    filename=filename,
                    status="rejected",
                    error="上传失败，请稍后重试",
                )
            )

    succeeded = sum(1 for item in results if item.status == "uploaded")
    return BatchUploadResponse(
        total=len(results),
        succeeded=succeeded,
        failed=len(results) - succeeded,
        results=results,
    )


@router.post(
    "/index/{doc_id}",
    response_model=IndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_document(
    doc_id: str,
    background_tasks: BackgroundTasks,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
    storage: FileStorage = Depends(get_file_storage),
):
    """建立索引（异步后台处理）。"""
    doc_info = kb.get_document_info(doc_id)
    if not doc_info:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"文档不存在: {doc_id}")

    if doc_info.status == DocumentStatus.COMPLETED:
        return IndexResponse(
            document_id=doc_id,
            status="already_indexed",
            total_chunks=doc_info.total_chunks,
            message="文档已索引",
        )

    if doc_info.status == DocumentStatus.PROCESSING:
        raise HTTPException(status.HTTP_409_CONFLICT, "文档正在处理中")

    if not storage.exists(doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文件不存在")

    def process() -> None:
        try:
            kb.build_index(doc_id, str(storage.path_for(doc_id)))
            logger.info(f"文档 {doc_id} 索引完成")
        except Exception as e:
            # 服务层已把状态标记为 failed，这里只需记录日志
            logger.exception(f"文档 {doc_id} 索引失败: {e}")

    background_tasks.add_task(process)

    return IndexResponse(
        document_id=doc_id,
        status="processing",
        total_chunks=0,
        message="索引任务已提交，正在后台处理",
    )


@router.post("/search/{doc_id}", response_model=SearchResponse)
async def search_document(
    doc_id: str,
    request: SearchRequest,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """在已索引文档中检索。"""
    doc_info = kb.get_document_info(doc_id)
    if not doc_info:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"文档不存在: {doc_id}")

    if doc_info.status != DocumentStatus.COMPLETED:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"文档尚未完成索引，当前状态: {doc_info.status.value}",
        )

    outcome = kb.search_with_confidence(
        doc_id, request.query, request.top_k
    )
    return SearchResponse(
        query=request.query,
        results=[ChunkResponse(**result) for result in outcome["results"]],
        total=len(outcome["results"]),
        confidence=outcome["confidence"],
        refused=outcome["refused"],
    )


def _answer_event_source(
    kb: KnowledgeBaseService, doc_id: str, query: str, top_k: int
):
    """把流式答案事件序列编码为 SSE data 行。"""
    try:
        for event in kb.stream_answer(doc_id, query, top_k):
            yield (
                "data: "
                f"{json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
            )
    except Exception as e:
        logger.exception(f"流式答案生成失败: {e}")
        payload = json.dumps(
            {"type": "error", "detail": "答案生成失败，请稍后重试"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        yield f"data: {payload}\n\n"


@router.post("/answer/{doc_id}", response_model=AnswerResponse)
def answer_document(
    doc_id: str,
    request: AnswerRequest,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
    stream: bool = Query(default=False),
):
    """检索并生成答案；stream=true 时以 SSE 流式返回。"""
    _require_completed_doc(kb, doc_id)

    if stream:
        return StreamingResponse(
            _answer_event_source(kb, doc_id, request.query, request.top_k),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = kb.answer(doc_id, request.query, request.top_k)
    except Exception as e:
        logger.exception(f"答案生成失败: {e}")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "答案生成失败，请稍后重试"
        )
    return AnswerResponse(**result)


def _group_answer_event_source(
    kb: KnowledgeBaseService, group_id: str, query: str, top_k: int
):
    """把分组流式答案事件序列编码为 SSE data 行。"""
    try:
        for event in kb.group_stream_answer(group_id, query, top_k):
            yield (
                "data: "
                f"{json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
            )
    except Exception as e:
        logger.exception(f"分组流式答案生成失败: {e}")
        payload = json.dumps(
            {"type": "error", "detail": "答案生成失败，请稍后重试"},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        yield f"data: {payload}\n\n"


@router.post("/groups", response_model=GroupInfo, status_code=status.HTTP_201_CREATED)
def create_group(
    request: GroupCreateRequest,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """创建命名分组。"""
    return kb.create_group(request.name)


@router.get("/groups", response_model=list[GroupInfo])
def list_groups(
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """列出全部分组。"""
    return kb.list_groups()


@router.get("/groups/{group_id}", response_model=GroupInfo)
def get_group(
    group_id: str,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """获取分组详情。"""
    group = kb.get_group(group_id)
    if not group:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"分组不存在: {group_id}")
    return group


@router.delete("/groups/{group_id}", response_model=DeleteResponse)
def delete_group(
    group_id: str,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """删除分组（成员文档标记为 pending）。"""
    if not kb.get_group(group_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"分组不存在: {group_id}")
    kb.delete_group(group_id)
    return DeleteResponse(status="deleted", document_id=group_id)


@router.post(
    "/groups/{group_id}/index/{doc_id}",
    response_model=IndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def index_document_into_group(
    group_id: str,
    doc_id: str,
    background_tasks: BackgroundTasks,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
    storage: FileStorage = Depends(get_file_storage),
):
    """把文档编入分组（后台处理）。"""
    group = kb.get_group(group_id)
    if not group:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"分组不存在: {group_id}")
    doc_info = kb.get_document_info(doc_id)
    if not doc_info:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"文档不存在: {doc_id}")
    if doc_info.status == DocumentStatus.PROCESSING:
        raise HTTPException(status.HTTP_409_CONFLICT, "文档正在处理中")
    if doc_id in group.doc_ids:
        return IndexResponse(
            document_id=doc_id,
            status="already_in_group",
            total_chunks=doc_info.total_chunks,
            message="文档已在分组中",
        )
    if not storage.exists(doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文件不存在")

    def process() -> None:
        try:
            kb.index_into_group(group_id, doc_id, str(storage.path_for(doc_id)))
            logger.info(f"文档 {doc_id} 编入分组 {group_id} 完成")
        except Exception as e:
            logger.exception(f"文档 {doc_id} 编入分组失败: {e}")

    background_tasks.add_task(process)
    return IndexResponse(
        document_id=doc_id,
        status="processing",
        total_chunks=0,
        message="编入分组任务已提交",
    )


@router.delete(
    "/groups/{group_id}/documents/{doc_id}",
    response_model=DeleteResponse,
)
def remove_document_from_group(
    group_id: str,
    doc_id: str,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """把文档移出分组（文档标记为 pending）。"""
    group = kb.get_group(group_id)
    if not group:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"分组不存在: {group_id}")
    if doc_id not in group.doc_ids:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"文档不在分组中: {doc_id}"
        )
    kb.group_remove_document(group_id, doc_id)
    return DeleteResponse(status="removed", document_id=doc_id)


@router.post("/groups/{group_id}/migrate", status_code=status.HTTP_202_ACCEPTED)
async def migrate_group(
    group_id: str,
    background_tasks: BackgroundTasks,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """把全部已索引文档迁移进分组（后台处理，不触发 OCR）。"""
    if not kb.get_group(group_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"分组不存在: {group_id}")

    def process() -> None:
        try:
            result = kb.migrate_group(group_id)
            logger.info(
                f"分组 {group_id} 迁移完成: "
                f"{result['succeeded']}/{result['total']}"
            )
        except Exception as e:
            logger.exception(f"分组 {group_id} 迁移失败: {e}")

    background_tasks.add_task(process)
    return {
        "status": "processing",
        "group_id": group_id,
        "message": "迁移任务已提交",
    }


@router.post("/groups/{group_id}/search", response_model=SearchResponse)
def search_group(
    group_id: str,
    request: SearchRequest,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """在分组共享集合中检索。"""
    if not kb.get_group(group_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"分组不存在: {group_id}")
    outcome = kb.group_search_with_confidence(
        group_id, request.query, request.top_k
    )
    return SearchResponse(
        query=request.query,
        results=[ChunkResponse(**result) for result in outcome["results"]],
        total=len(outcome["results"]),
        confidence=outcome["confidence"],
        refused=outcome["refused"],
    )


@router.post("/groups/{group_id}/answer", response_model=AnswerResponse)
def answer_group(
    group_id: str,
    request: AnswerRequest,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
    stream: bool = Query(default=False),
):
    """分组检索并生成答案；stream=true 时 SSE 流式。"""
    if not kb.get_group(group_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"分组不存在: {group_id}")
    if stream:
        return StreamingResponse(
            _group_answer_event_source(
                kb, group_id, request.query, request.top_k
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    try:
        result = kb.group_answer(group_id, request.query, request.top_k)
    except Exception as e:
        logger.exception(f"分组答案生成失败: {e}")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "答案生成失败，请稍后重试"
        )
    return AnswerResponse(**result)


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents(
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """列出所有文档。"""
    return kb.list_documents()


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """获取文档信息。"""
    doc = kb.get_document_info(doc_id)
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    return doc


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
    storage: FileStorage = Depends(get_file_storage),
):
    """删除文档、索引与文件。"""
    if not kb.get_document_info(doc_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")

    kb.delete_document(doc_id)
    storage.delete(doc_id)

    return DeleteResponse(status="deleted", document_id=doc_id)


@router.get("/stats/{doc_id}", response_model=StatsResponse)
async def get_stats(
    doc_id: str,
    kb: KnowledgeBaseService = Depends(get_knowledge_base),
):
    """获取索引统计。"""
    engine = kb.get_engine(doc_id)
    if not engine:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    return engine.get_stats()
