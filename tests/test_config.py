import pytest
from pydantic import ValidationError

from src.core.config import Settings

SECRET_ENV = {
    "QWEN_API_KEY": "k",
    "QWEN_BASE_URL": "https://example.com/v1",
    "PADDLEOCR_API_KEY": "p",
}


def test_loads_new_env_names(monkeypatch):
    for key, value in SECRET_ENV.items():
        monkeypatch.setenv(key, value)
    settings = Settings(_env_file=None)
    assert settings.qwen_api_key == "k"
    assert settings.qwen_base_url == "https://example.com/v1"
    assert settings.paddleocr_api_key == "p"


def test_legacy_env_aliases(monkeypatch):
    # 确保新命名不存在，验证旧命名兼容
    for key in ("QWEN_API_KEY", "QWEN_BASE_URL", "PADDLEOCR_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("QWEN3_API_KEY", "k3")
    monkeypatch.setenv("QWEN3_BASE_URL", "https://example.com/v1")
    monkeypatch.setenv("PADLLEOCR_API_KEY", "p")
    monkeypatch.setenv("QWEM3_MODLE_NAME", "legacy-model")
    settings = Settings(_env_file=None)
    assert settings.qwen_api_key == "k3"
    assert settings.qwen_embedding_model == "legacy-model"


def test_missing_secret_raises(monkeypatch):
    monkeypatch.delenv("QWEN_API_KEY", raising=False)
    monkeypatch.delenv("QWEN3_API_KEY", raising=False)
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    monkeypatch.delenv("QWEN3_BASE_URL", raising=False)
    monkeypatch.delenv("PADDLEOCR_API_KEY", raising=False)
    monkeypatch.delenv("PADLLEOCR_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_max_upload_size_alias(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_SIZE", "42")
    settings = Settings(_env_file=None, **SECRET_ENV)
    assert settings.max_upload_size_mb == 42


def test_cors_origin_list():
    settings = Settings(
        _env_file=None,
        **SECRET_ENV,
        cors_origins="http://a.com, http://b.com",
    )
    assert settings.cors_origin_list == ["http://a.com", "http://b.com"]


def test_paddleocr_base_url_suffix_normalized(monkeypatch):
    monkeypatch.setenv(
        "PADDLEOCR_BASE_URL",
        "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
    )
    settings = Settings(_env_file=None, **SECRET_ENV)
    assert settings.paddleocr_base_url == "https://paddleocr.aistudio-app.com"


def test_invalid_log_level_rejected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **SECRET_ENV)


def test_retrieval_and_confidence_defaults(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "25")
    settings = Settings(_env_file=None, **SECRET_ENV)
    assert settings.retrieval_candidate_k == 25
    assert settings.bm25_enabled is True
    assert settings.chapter_query_routing is True
    assert settings.answer_confidence_threshold == 0.55
    assert settings.subchunk_enabled is True
    assert settings.subchunk_max_tokens == 160
