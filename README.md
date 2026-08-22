# 数学 RAG 服务

基于 **PaddleOCR-VL + Qwen Embedding + ChromaDB** 的 PDF 智能问答系统：
上传 PDF → PaddleOCR-VL 解析为结构化内容块 → Qwen 生成向量 → ChromaDB
持久化索引 → 语义检索。

## 功能

- PDF 上传与异步索引（`BackgroundTasks`）
- 文档状态机：`pending → processing → completed / failed`
- 向量检索（余弦相似度），返回内容块与页码
- 文档元数据持久化（JSON 注册表），重启不丢
- 文档/索引删除时同步清理 Chroma 集合
- 可配置日志（loguru，按天轮转）
- 结构化 API 错误与统一校验响应

## 项目结构

```text
.
├── src/
│   ├── main.py               # FastAPI 入口
│   ├── run.py                # 开发服务器（--reload）
│   ├── api/
│   │   ├── routes.py         # REST 路由
│   │   └── dependencies.py   # 依赖注入（单例）
│   ├── core/
│   │   ├── config.py         # pydantic-settings 配置
│   │   ├── models.py         # Pydantic 数据模型
│   │   ├── parser.py         # PaddleOCR-VL 解析器
│   │   ├── embedding.py      # Qwen Embedding（批处理+重试）
│   │   ├── indexer.py        # ChromaDB 向量索引器
│   │   └── rag_engine.py     # 解析→向量→索引协调引擎
│   ├── services/
│   │   └── knowledge_base.py # 知识库服务（注册表+生命周期）
│   └── utils/
│       ├── logger.py         # loguru 配置
│       └── helpers.py        # 通用工具
├── scripts/
│   └── check_embedding_models.py
├── tests/                    # pytest 测试
├── pyproject.toml
└── .env.example
```

## 快速开始

要求：Python 3.14+，[uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装依赖
uv sync --dev

# 2. 配置环境变量
cp .env.example .env
# 填入 QWEN_API_KEY / PADDLEOCR_API_KEY 等

# 3. 启动服务
uv run rag-api            # 或 uv run python -m src.main
# 开发模式（热重载）：
uv run python -m src.run
```

打开 http://localhost:8000/docs 查看 Swagger 文档。
打开 http://localhost:8000 使用内置前端控制台（上传、索引、检索、删除）。

## API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/upload` | 上传 PDF，返回文档信息（201） |
| POST | `/api/v1/index/{doc_id}` | 提交异步索引任务（202） |
| POST | `/api/v1/search/{doc_id}` | 在文档中检索 |
| GET | `/api/v1/documents` | 列出全部文档 |
| GET | `/api/v1/documents/{doc_id}` | 文档详情 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档+索引+文件 |
| GET | `/api/v1/stats/{doc_id}` | 索引统计 |
| GET | `/health` | 健康检查 |
| GET | `/` | 前端控制台页面 |

## 环境变量

完整列表见 [.env.example](.env.example)。兼容旧的拼写错误键名
（如 `QWEN3_API_KEY`、`PADLLEOCR_API_KEY`、`QWEM3_MODLE_NAME`、
`QWEN3_BASE_URLL`），推荐迁移到新命名。

## 测试与代码质量

```bash
uv run pytest               # 运行测试
uv run ruff check .         # 代码检查
uv run ruff format .        # 代码格式化
```

## 说明

- `src/dots-ocr/` 为早期原型脚本，已由 `src/core/parser.py` 替代，
  确认无用后可删除。
- 生产环境建议将任务队列替换为 Celery / RQ 等持久化队列，
  并使用反向代理 + HTTPS 对外提供服务。
