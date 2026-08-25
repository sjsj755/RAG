"""RAG API 路由。"""

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from src.api.dependencies import get_app_settings, get_file_storage, get_knowledge_base
from src.core.config import Settings
from src.core.models import (
    BatchUploadItem,
    BatchUploadResponse,
    ChunkResponse,
    DeleteResponse,
    DocumentInfo,
    DocumentStatus,
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
