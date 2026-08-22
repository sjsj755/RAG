"""pytest 全局配置：在导入应用前设置测试环境变量。"""

import os

os.environ.setdefault("QWEN_API_KEY", "test-qwen-key")
os.environ.setdefault("QWEN_BASE_URL", "https://example.com/v1")
os.environ.setdefault("PADDLEOCR_API_KEY", "test-paddle-key")
os.environ.setdefault("PADDLEOCR_BASE_URL", "https://example.com/ocr")
os.environ.setdefault("UPLOAD_DIR", "test_uploads")
os.environ.setdefault("CHROMA_PERSIST_DIR", "test_chroma")
os.environ.setdefault("LOG_DIR", "test_logs")
os.environ.setdefault("REGISTRY_FILE", "test_registry.json")
