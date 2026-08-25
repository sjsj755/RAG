from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def to_dict(obj: Any) -> Any:
    """递归将对象转换为纯 Python 字典。"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "dict"):
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return {
            k: to_dict(v)
            for k, v in vars(obj).items()
            if not k.startswith("_")
        }
    else:
        return obj


def generate_document_id(filename: str) -> str:
    """根据时间戳与随机熵生成唯一 ID。"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_part = uuid4().hex[:8]
    return f"doc_{timestamp}_{unique_part}"


def ensure_dir(path: str) -> Path:
    """确保目录存在，返回目录对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def compute_confidence(
    max_cosine: float, agree_legs: int, leg_count: int
) -> float:
    """检索置信度 = 最大余弦 × (0.5 + 0.5 × 多路一致比例)。

    单路检索时一致比例为 1，置信度退化为最大余弦；无向量结果时为 0。
    """
    if max_cosine <= 0 or leg_count <= 0:
        return 0.0
    ratio = agree_legs / leg_count if leg_count else 0.0
    return max_cosine * (0.5 + 0.5 * ratio)
