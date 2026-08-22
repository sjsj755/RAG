"""检查 Qwen 兼容接口当前可用的 embedding 模型。

用法: uv run python scripts/check_embedding_models.py
"""

from openai import OpenAI

from src.core.config import settings
from src.utils.logger import logger


def check_available_models() -> str | None:
    """依次尝试候选模型，返回第一个可用的模型名。"""
    client = OpenAI(
        api_key=settings.qwen_api_key,
        base_url=settings.qwen_base_url,
    )
    candidates = [
        settings.qwen_embedding_model,
        "text-embedding-v4",
        "qwen-embedding",
        "text-embedding-v3",
    ]

    for model in candidates:
        try:
            client.embeddings.create(model=model, input=["测试文本"])
            logger.info(f"模型 {model} 可用")
            return model
        except Exception as e:
            logger.warning(f"模型 {model} 不可用: {e}")
    return None


if __name__ == "__main__":
    result = check_available_models()
    if result:
        logger.success(f"推荐使用模型: {result}")
    else:
        logger.error("未找到可用模型，请检查 QWEN_API_KEY / QWEN_BASE_URL")
