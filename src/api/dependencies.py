from src.services.knowledge_base import KnowledgeBaseService

# 单例
_kb_service = None

def get_knowledge_base() -> KnowledgeBaseService:
    global _kb_service
    if _kb_service is None:
        _kb_service = KnowledgeBaseService()
    return _kb_service
