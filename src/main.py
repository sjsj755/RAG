"""FastAPI 应用入口。"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import router
from src.core.config import settings
from src.utils.logger import logger

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    logger.info("启动 RAG 服务...")
    logger.info(f"上传目录: {settings.upload_dir}")
    logger.info(f"Chroma 持久化目录: {settings.chroma_persist_dir}")
    yield
    logger.info("关闭 RAG 服务...")


app = FastAPI(
    title="多模态 RAG API",
    description="基于 PaddleOCR-VL + Qwen 的 PDF 智能问答系统",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：允许来源可配置；使用通配符时不允许携带凭据
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=settings.cors_origins != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)

# 静态资源（前端控制台）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
async def health_check():
    """健康检查。"""
    return {"status": "healthy", "service": "multimodal-rag"}


@app.get("/", include_in_schema=False)
async def root():
    """返回前端控制台页面。"""
    return FileResponse(STATIC_DIR / "index.html")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """参数校验失败时返回结构化错误。"""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "code": "validation_error"},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """兜底异常处理：记录详情，但不向客户端泄露内部信息。"""
    logger.exception(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "code": "internal_error"},
    )


def main() -> None:
    """开发服务器入口（uvicorn）。"""
    import uvicorn

    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
