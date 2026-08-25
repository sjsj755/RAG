"""应用配置。

所有配置统一从环境变量 / .env 文件加载，集中在此处管理，
避免在业务代码中散落 os.getenv 调用。

注意：为兼容旧 .env 中拼写有误的键名，部分字段通过 AliasChoices
同时接受新旧两种命名，推荐迁移到新命名（见 .env.example）。
"""

from functools import lru_cache

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """全局应用配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 应用 ----------
    app_name: str = "multimodal-rag"
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(
        default="INFO",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
    )
    max_upload_size_mb: int = Field(
        default=100,
        ge=1,
        le=2048,
        validation_alias=AliasChoices("MAX_UPLOAD_SIZE_MB", "MAX_UPLOAD_SIZE"),
    )
    # 单次批量上传的文件数上限
    max_batch_files: int = Field(default=20, ge=1, le=100)

    # ---------- 目录 ----------
    upload_dir: str = Field(default="uploads")
    chroma_persist_dir: str = Field(default="chroma_db")
    log_dir: str = Field(default="logs")
    registry_file: str = Field(default="registry.json")

    # ---------- CORS ----------
    # 逗号分隔的允许来源列表，例如: http://localhost:5173,http://localhost:3000
    cors_origins: str = Field(default="*")

    # ---------- PaddleOCR ----------
    paddleocr_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("PADDLEOCR_API_KEY", "PADLLEOCR_API_KEY"),
    )
    paddleocr_base_url: str | None = Field(
        default="https://paddleocr.aistudio-app.com",
        validation_alias=AliasChoices("PADDLEOCR_BASE_URL", "PADLLEOCR_BASE_URL"),
    )
    paddleocr_model: str = Field(
        default="PaddleOCR-VL-1.6",
        validation_alias=AliasChoices("PADDLEOCR_MODEL", "PADLLEOCR_MODLE_NAME"),
    )

    # ---------- Qwen ----------
    qwen_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("QWEN_API_KEY", "QWEN3_API_KEY"),
    )
    qwen_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("QWEN_BASE_URL", "QWEN3_BASE_URL"),
    )
    qwen_embedding_model: str = Field(
        default="text-embedding-v4",
        validation_alias=AliasChoices(
            "QWEN_EMBEDDING_MODEL", "QWEM3_MODLE_NAME"
        ),
    )

    # ---------- DeepSeek（查询改写/补全） ----------
    deepseek_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "DEEPSEEK_KEY"),
    )
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # ---------- 分块 ----------
    chunk_max_tokens: int = Field(default=512, ge=64, le=4096)
    # 超长块滑动窗口的重叠比例（0 ~ 0.5）
    chunk_overlap_ratio: float = Field(default=0.15, ge=0.0, lt=0.5)

    # ---------- 查询改写 ----------
    query_rewrite_enabled: bool = True
    query_rewrite_count: int = Field(default=2, ge=1, le=4)
    # 改写查询与原始查询的相似度阈值，低于阈值丢弃
    query_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)

    # ---------- 知识图谱 ----------
    kg_enabled: bool = True

    # ---------- 检索 ----------
    # 每路召回候选深度（RRF 融合后再按请求 top_k 返回）
    retrieval_candidate_k: int = Field(default=20, ge=5, le=100)
    # BM25 稀疏检索路开关
    bm25_enabled: bool = True
    # 章节/目录类查询路由：只保留同义改写并加入原始查询
    chapter_query_routing: bool = True
    # 低置信度拒绝阈值（0-1，设为 0 关闭拒绝）
    answer_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    # LLM 二阶段重排（仅概念/定义类查询）
    llm_rerank_enabled: bool = True
    llm_rerank_top_n: int = Field(default=20, ge=5, le=50)
    # LLM 答案生成
    answer_temperature: float = Field(default=0.3, ge=0.0, le=1.5)
    answer_max_tokens: int = Field(default=1024, ge=64, le=8192)

    # ---------- 子块 ----------
    subchunk_enabled: bool = True
    subchunk_max_tokens: int = Field(default=160, ge=64, le=256)

    # ---------- RAG 内部参数 ----------
    embedding_batch_size: int = Field(default=10, ge=1, le=64)
    default_top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _check_required_secrets(self) -> Settings:
        """关键密钥缺失时快速失败，避免运行到一半才报错。"""
        missing = [
            name
            for name, value in (
                ("QWEN_API_KEY", self.qwen_api_key),
                ("QWEN_BASE_URL", self.qwen_base_url),
                ("PADDLEOCR_API_KEY", self.paddleocr_api_key),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"缺少必需的环境变量: {', '.join(missing)}。"
                "请参考 .env.example 配置 .env 文件。"
            )
        return self

    @field_validator("paddleocr_base_url")
    @classmethod
    def _normalize_paddleocr_base_url(
        cls, value: str | None
    ) -> str | None:
        """SDK 会自动拼接 /api/v2/ocr/jobs，兼容配置里已带完整路径的情况。"""
        if not value:
            return value
        suffix = "/api/v2/ocr/jobs"
        value = value.rstrip("/")
        if value.endswith(suffix):
            value = value[: -len(suffix)]
        return value

    @property
    def cors_origin_list(self) -> list[str]:
        """解析逗号分隔的 CORS 来源为列表。"""
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例（带缓存，便于测试时重置）。"""
    return Settings()


settings = get_settings()
