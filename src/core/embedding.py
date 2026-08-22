import time

from openai import OpenAI

from src.core.config import Settings, get_settings
from src.utils.logger import logger


class EmbeddingGenerator:
    def __init__(self, config: Settings | None = None) -> None:
        cfg = config or get_settings()
        self.client = OpenAI(
            api_key=cfg.qwen_api_key,
            base_url=cfg.qwen_base_url,
            timeout=60.0,
        )
        self.model = cfg.qwen_embedding_model
        self.batch_size = cfg.embedding_batch_size

    def embed(self, texts: list[str], max_retries: int = 3) -> list[list[float]]:
        """批量生成向量，带重试机制"""
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, self.batch_size):
            batch = texts[i:i + self.batch_size]
            for attempt in range(max_retries):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch,
                    )
                    embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(embeddings)
                    logger.debug(f"嵌入批次 {i // self.batch_size + 1} 完成")
                    break
                except Exception as e:
                    logger.warning(f"嵌入失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # 指数退避
                    else:
                        raise RuntimeError(f"嵌入失败: {e}")

        logger.info(f"嵌入完成，共 {len(all_embeddings)} 个向量")
        return all_embeddings
