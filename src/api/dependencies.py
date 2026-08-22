"""FastAPI 依赖注入：应用装配点。"""

from fastapi import Depends

from src.core.config import Settings, get_settings
from src.services.file_storage import FileStorage
from src.services.knowledge_base import KnowledgeBaseService


def get_app_settings() -> Settings:
    """应用配置（composition root）。"""
    return get_settings()


# 单例：知识库服务持有引擎与注册表状态
_kb_service: KnowledgeBaseService | None = None


def get_knowledge_base(
    settings: Settings = Depends(get_app_settings),
) -> KnowledgeBaseService:
    """知识库服务单例。"""
    global _kb_service
    if _kb_service is None:
        _kb_service = KnowledgeBaseService(config=settings)
    return _kb_service


def get_file_storage(
    settings: Settings = Depends(get_app_settings),
) -> FileStorage:
    """文件存储（无状态，按配置即时创建）。"""
    return FileStorage(settings.upload_dir)
