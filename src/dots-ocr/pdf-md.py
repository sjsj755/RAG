import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

from paddleocr import PaddleOCRClient
from openai import OpenAI
import chromadb
from chromadb.config import Settings

load_dotenv(override=True)


class MathPDFRAG:
    def __init__(self):
        load_dotenv(override=True)

        self.ocr_client = PaddleOCRClient(
            token=os.getenv("padlleocr_api_key"),
        )

        self.embed_client = OpenAI(
            api_key=os.getenv("qwen3_api_key"),
            base_url=os.getenv("qwen3_base_url"),
        )
        self.embed_model = os.getenv("QWEM3_MODLE_NAME")

        self.chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        self.collection_name = "math_pdf_chunks"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def parse_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """调用 PaddleOCR-VL-1.6 API 解析 PDF"""
        print("📤 提交解析任务...")
        result = self.ocr_client.parse_document(
            file_path=pdf_path,
            model="PaddleOCR-VL-1.6"
        )
        print("✅ 解析完成")
        return result

    def _to_dict(self, obj):
        """递归将对象转换为字典"""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: self._to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._to_dict(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return {k: self._to_dict(v) for k, v in vars(obj).items() if not k.startswith('_')}
        else:
            return obj

    def extract_chunks(self, parse_result: Any) -> List[Dict[str, Any]]:
        """从解析结果中提取文本块（适配新版返回结构）"""
        # 将结果转为字典
        parse_result = self._to_dict(parse_result)
        pages = parse_result.get('pages', [])
        chunks = []

        for page in pages:
            page_num = page.get('page_num', 0)
            pruned = page.get('pruned_result')
            if pruned:
                parsing_res_list = pruned.get('parsing_res_list', [])
                for block in parsing_res_list:
                    block_label = block.get('block_label', '')
                    block_content = block.get('block_content', '').strip()
                    # 跳过无内容或不需要的标签（可根据需要调整）
                    if block_content and block_label not in ['header', 'footer', 'header_image', 'footer_image']:
                        chunks.append({
                            'text': block_content,
                            'type': block_label,  # 例如 'doc_title', 'text', 'display_formula'
                            'page_num': page_num,
                            'block_id': block.get('block_id', '')
                        })
            else:
                # 降级方案：如果没有 pruned_result，则使用整体 markdown_text
                markdown_text = page.get('markdown_text', '').strip()
                if markdown_text:
                    chunks.append({
                        'text': markdown_text,
                        'type': 'markdown',
                        'page_num': page_num,
                        'block_id': f'page_{page_num}'
                    })

        return chunks

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """调用 text-embedding-v4 API 批量生成向量"""
        print(f"🔍 使用的模型名称: {repr(self.embed_model)}")
        print(f"🔍 使用的 Base URL: {self.embed_client.base_url}")
        batch_size = 10
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"🔍 正在嵌入第 {i // batch_size + 1} 批 ({len(batch)} 条)...")

            response = self.embed_client.embeddings.create(
                model=self.embed_model,  # 此时为 "text-embedding-v4"
                input=batch,  # 直接传字符串列表
                # 可选：指定维度，默认为 1536
                # dimensions=1536,
            )
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    def build_index(self, chunks: List[Dict[str, Any]]):
        """将文本块向量化并存入 Chroma"""
        if not chunks:
            print("⚠️ 没有提取到任何文本块")
            return

        texts = [chunk['text'] for chunk in chunks]
        print(f"📄 共 {len(texts)} 个内容块，开始生成向量...")

        # 批量生成向量
        embeddings = self.embed_texts(texts)

        # 准备 Chroma 数据
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        metadatas = []
        for chunk in chunks:
            meta = {k: v for k, v in chunk.items() if k != 'text'}
            metadatas.append(meta)

        # 存入 Chroma（清空旧数据，避免重复）
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except:
            pass
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        print(f"✅ 索引构建完成，共 {self.collection.count()} 个向量")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """检索与查询最相关的 top_k 个文本块"""
        if self.collection.count() == 0:
            raise RuntimeError("索引为空，请先调用 build_index")

        # 生成查询向量
        query_embedding = self.embed_texts([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                chunk = {
                    'text': results['documents'][0][i],
                    'score': 1 - results['distances'][0][i],
                }
                chunk.update(results['metadatas'][0][i])
                retrieved.append(chunk)
        return retrieved


def main():
    # ---------- 配置 ----------
    pdf_file ="D:/pycharm/RAG/src/data/a/output_part1.pdf"      # 替换为你的PDF路径

    # ---------- 初始化 RAG ----------
    rag = MathPDFRAG()

    # 1. 解析 PDF
    parse_result = rag.parse_pdf(pdf_file)

    # --- 添加以下调试代码 ---
    print(f"任务状态: {getattr(parse_result, 'status', 'N/A')}")
    print(f"任务错误信息: {getattr(parse_result, 'task_error', 'N/A')}")
    print(f"结果对象属性: {dir(parse_result)}")
    # 尝试获取原始响应
    if hasattr(parse_result, 'raw_response'):
        print(f"原始响应: {parse_result.raw_response}")
    # ------------------------

    # 2. 提取文本块
    chunks = rag.extract_chunks(parse_result)
    print(f"📄 共提取 {len(chunks)} 个内容块")

    # 3. 构建索引
    rag.build_index(chunks)

    # 4. 交互式检索
    print("\n💡 输入问题（输入 'quit' 退出）：")
    while True:
        query = input("> ")
        if query.lower() in ('quit', 'exit'):
            break
        results = rag.search(query, top_k=3)
        print(f"\n📌 找到 {len(results)} 个相关片段：")
        for i, res in enumerate(results, 1):
            print(f"\n--- 结果 {i} (页码: {res.get('page_num')}, 类型: {res.get('type')}) ---")
            print(res['text'][:300] + ("..." if len(res['text']) > 300 else ""))


if __name__ == "__main__":
    main()