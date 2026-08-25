"""从现有 KG 数据重建子块索引（不触发 OCR）。

用法:
    python scripts/rebuild_index.py --doc-id <doc_id>

读取 <chroma_dir>/<doc_id>.kg.json 中的父块 → 子块化 + 公式规范化 →
Qwen 嵌入 → 重建该文档的 Chroma 集合与 KG 文件，并更新注册表块数。
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from src.core.chunker import SemanticChunker
from src.core.config import get_settings
from src.core.embedding import EmbeddingGenerator
from src.core.indexer import VectorIndexer
from src.core.knowledge_graph import RuleKnowledgeGraph
from src.core.models import DocumentStatus
from src.services.document_repository import DocumentRepository
from src.utils.logger import logger
from src.utils.math_normalize import normalize_math_text


def main() -> None:
    parser = argparse.ArgumentParser(description="重建子块索引（不重新 OCR）")
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--chroma-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--subchunk-max-tokens", type=int, default=None)
    args = parser.parse_args()

    cfg = get_settings()
    chroma_dir = Path(args.chroma_dir or cfg.chroma_persist_dir)
    kg_path = chroma_dir / f"{args.doc_id}.kg.json"
    if not kg_path.exists():
        raise SystemExit(
            f"KG 文件不存在: {kg_path}；请先完成正常索引流程"
        )

    data = json.loads(kg_path.read_text(encoding="utf-8"))
    parent_chunks = list(data.get("chunks", {}).values())
    if not parent_chunks:
        raise SystemExit(f"KG 文件中没有内容块: {kg_path}")

    chunker = SemanticChunker(
        max_tokens=cfg.chunk_max_tokens,
        overlap_ratio=cfg.chunk_overlap_ratio,
    )
    subchunks = chunker.build_subchunks(
        parent_chunks,
        args.subchunk_max_tokens or cfg.subchunk_max_tokens,
    )
    for sub in subchunks:
        sub["text"] = normalize_math_text(sub["text"])

    logger.info(
        f"子块化完成: {len(parent_chunks)} 个父块 → {len(subchunks)} 个子块"
    )
    embedder = EmbeddingGenerator(config=cfg)
    texts = [sub["text"] for sub in subchunks]
    embeddings = embedder.embed(texts)

    indexer = VectorIndexer(collection_name=args.doc_id, config=cfg)
    indexer.clear()
    indexer.add(subchunks, embeddings)

    kg = RuleKnowledgeGraph()
    kg.build(parent_chunks)
    kg.save(kg_path)

    repo = DocumentRepository(cfg.registry_file)
    docs = repo.load()
    doc = docs.get(args.doc_id)
    if doc is not None:
        doc.status = DocumentStatus.COMPLETED
        doc.total_chunks = len(subchunks)
        doc.updated_at = datetime.now(UTC)
        repo.save(docs)
        logger.info(f"注册表已更新: {args.doc_id} total_chunks={len(subchunks)}")
    else:
        logger.warning(f"注册表中不存在 {args.doc_id}，仅重建索引")

    print(f"重建完成: {len(subchunks)} 个子块已写入集合 {args.doc_id}")


if __name__ == "__main__":
    main()
