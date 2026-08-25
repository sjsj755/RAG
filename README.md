# 数学 RAG 服务

基于 **PaddleOCR-VL + Qwen Embedding + ChromaDB** 的 PDF 智能问答系统：
上传 PDF → PaddleOCR-VL 解析为结构化内容块 → 语义分块/子块 → Qwen 生成向量 →
ChromaDB 持久化索引 → 多路融合检索 → LLM 重排 → 置信度门控。

## 功能

- PDF 上传：单文件与批量上传（`.pdf` 魔数校验、空文件拒绝、大小限制、单文件失败隔离）
- 异步索引（`BackgroundTasks`），文档状态机：`pending → processing → completed / failed`
- LLM 答案生成：DeepSeek 基于检索片段生成带 `[n]` 引用的答案，支持 JSON 与 SSE 流式，
  低置信度时直接拒绝（不调用 LLM）
- 分组联合索引：命名分组共享一个 Chroma 集合，多文档联合检索/问答，支持存量迁移
- 检索链路：
  - DeepSeek 查询改写（CARE 提示词：同义改写 + 关键词扩展，embedding 相似度过滤）
  - 章节/目录类查询路由（只保留同义改写，并加入原始查询）
  - 多路召回：向量检索 + 规则知识图谱 + BM25 稀疏检索，RRF 融合
  - 概念/定义类查询的 DeepSeek 二阶段重排（失败自动回退原顺序）
- 子块化索引：父块按句子切成约 160 token 子块嵌入；命中子块时返回完整父块，
  并附命中片段 `fragment` 供前端优先展示
- 低置信度拒绝：置信度 = 最大余弦 × 多路一致比例，低于阈值时返回
  `refused=true`（结果保留，前端折叠展示并提示人工判断）
- 文档元数据 JSON 注册表持久化，删除时同步清理 Chroma 集合与知识图谱文件
- 可配置日志（loguru，按天轮转）、结构化 API 错误与统一校验响应
- 检索评测：30 题标注集自动计算 Recall@5 / MRR@5 / NDCG@5 / Precision@5 /
  top1 相关率 / 无答案拒答率，支持改写开关对比与人工复核抽样

## 项目结构

```text
.
├── src/
│   ├── main.py               # FastAPI 入口
│   ├── run.py                # 开发服务器（--reload）
│   ├── api/
│   │   ├── routes.py         # REST 路由
│   │   └── dependencies.py   # 依赖注入（配置/服务/存储装配点）
│   ├── core/
│   │   ├── config.py         # pydantic-settings 配置
│   │   ├── models.py         # Pydantic 数据模型
│   │   ├── parser.py         # PaddleOCR-VL 解析器
│   │   ├── chunker.py        # 语义分块 + 子块化
│   │   ├── embedding.py      # Qwen Embedding（批处理+重试）
│   │   ├── indexer.py        # ChromaDB 向量索引器
│   │   ├── bm25_retriever.py # BM25 稀疏检索路
│   │   ├── knowledge_graph.py# 规则知识图谱（概念→块映射）
│   │   ├── fusion.py         # RRF 多路融合
│   │   ├── query_rewriter.py # DeepSeek 查询改写（CARE 提示词）
│   │   ├── reranker.py       # LLM 二阶段重排
│   │   ├── protocols.py      # 组件接口（依赖注入用 Protocol）
│   │   └── rag_engine.py     # 检索编排引擎（多路召回/重排/置信度）
│   ├── services/
│   │   ├── knowledge_base.py     # 知识库服务（注册表+生命周期）
│   │   ├── document_repository.py # 文档注册表持久化
│   │   └── file_storage.py        # PDF 文件存储（原子写入）
│   └── utils/
│       ├── logger.py         # loguru 配置（启动时装配）
│       ├── helpers.py        # 通用工具
│       ├── eval_metrics.py   # 检索评测指标与抽样
│       ├── math_normalize.py # 公式文本规范化
│       ├── math_terms.py     # 高中数学概念术语表
│       └── query_routing.py  # 章节/概念查询路由
├── scripts/
│   ├── evaluate_rag.py       # 30 题检索评测（自动指标 + 人工复核抽样）
│   ├── compare_eval.py       # 两组评测报告对比
│   ├── rebuild_index.py      # 从现有 KG 重建子块索引（不触发 OCR）
│   ├── eval_questions.json   # 30 题标注评测集
│   └── check_embedding_models.py
├── tests/                    # pytest 测试（124 个）
├── pyproject.toml
├── uv.lock
└── .env.example
```

## 快速开始

要求：Python 3.14+，[uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装依赖（含 rank-bm25）
uv sync --dev

# 2. 配置环境变量
cp .env.example .env
# 填入 QWEN_API_KEY / PADDLEOCR_API_KEY / DEEPSEEK_API_KEY 等

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
| POST | `/api/v1/upload` | 上传单个 PDF，返回文档信息（201） |
| POST | `/api/v1/upload/batch` | 批量上传（串行处理，单文件失败不影响其他，200+逐项状态） |
| POST | `/api/v1/index/{doc_id}` | 提交异步索引任务（202） |
| POST | `/api/v1/search/{doc_id}` | 检索；响应含 `confidence`、`refused`，结果项含 `fragment` |
| POST | `/api/v1/answer/{doc_id}` | 检索并生成答案；`?stream=true` 时 SSE 流式，低置信直接拒绝 |
| POST | `/api/v1/groups` | 创建分组（201） |
| GET | `/api/v1/groups` | 列出全部分组 |
| GET | `/api/v1/groups/{id}` | 分组详情 |
| DELETE | `/api/v1/groups/{id}` | 删除分组（成员文档标记为 pending） |
| POST | `/api/v1/groups/{id}/index/{doc_id}` | 把文档编入分组（后台） |
| DELETE | `/api/v1/groups/{id}/documents/{doc_id}` | 把文档移出分组（标记 pending） |
| POST | `/api/v1/groups/{id}/migrate` | 把全部已索引文档迁移进分组（后台，不触发 OCR） |
| POST | `/api/v1/groups/{id}/search` | 分组内联合检索（结果带 doc_id/filename） |
| POST | `/api/v1/groups/{id}/answer` | 分组内问答；`?stream=true` 时 SSE 流式 |
| GET | `/api/v1/documents` | 列出全部文档 |
| GET | `/api/v1/documents/{doc_id}` | 文档详情 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档+索引+文件 |
| GET | `/api/v1/stats/{doc_id}` | 索引统计 |
| GET | `/health` | 健康检查 |
| GET | `/` | 前端控制台页面 |

## 检索流程

```text
用户查询
  → 公式文本规范化
  → 查询改写（DeepSeek，CARE 提示词，相似度过滤）
  → 章节类查询：只保留同义改写 + 原始查询；其余：改写替代原始
  → 多路召回：向量（每个查询变体） + 知识图谱 + BM25（原始查询）
  → RRF 融合（k=60，按父块去重）
  → 概念/定义类查询：DeepSeek 重排（失败回退）
  → 置信度 = 最大余弦 × (0.5 + 0.5 × 多路一致比例)
  → 低于阈值标记 refused（结果仍返回，供前端人工判断）
```

## 答案生成 API

在检索之上调用 DeepSeek 生成答案；检索置信度低于 `ANSWER_CONFIDENCE_THRESHOLD`
时直接拒绝，不调用 LLM。

### 端点与参数

```text
POST /api/v1/answer/{doc_id}?stream=false|true
```

请求体：

```json
{"query": "什么是集合？", "top_k": 5}
```

`stream=false`（默认）一次性返回 JSON；`stream=true` 返回 SSE 流式。

### 非流式响应（成功）

```json
{
  "query": "什么是集合？",
  "answer": "集合是把一些元素组成的总体叫做集合（set）……[1]",
  "refused": false,
  "refusal_reason": null,
  "confidence": 0.75,
  "sources": [
    {"index": 1, "page_num": 9, "text": "一般地，我们把研究对象统称为元素……", "score": 0.78}
  ]
}
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `query` | 原始问题 |
| `answer` | 生成的答案（公式为 LaTeX，引用用 `[n]` 标注） |
| `refused` | 是否因低置信度拒绝 |
| `refusal_reason` | 拒绝原因，未拒绝时为 `null` |
| `confidence` | 检索置信度（0-1） |
| `sources` | 引用来源数组：`index` 编号、`page_num` 页码、`text` 命中片段/父块文本、`score` 相关度 |

### 非流式响应（低置信拒绝）

```json
{
  "query": "量子纠缠是什么？",
  "answer": null,
  "refused": true,
  "refusal_reason": "检索置信度不足，未生成答案",
  "confidence": 0.45,
  "sources": []
}
```

### SSE 流式事件

每行 `data: {...}`，事件之间用空行分隔；正常顺序为 `sources → answer×n → done`，
低置信为 `refused → done`，生成失败为 `error`：

```text
data: {"type":"sources","sources":[{"index":1,"page_num":9,"text":"……","score":0.78}],"confidence":0.75}

data: {"type":"answer","content":"集合"}

data: {"type":"answer","content":"是把一些元素组成的总体……"}

data: {"type":"done","refused":false,"confidence":0.75}
```

事件类型：

| 类型 | 说明 |
| --- | --- |
| `sources` | 引用来源列表与置信度（先于答案发出） |
| `answer` | 答案片段，前端逐段拼接 |
| `refused` | 低置信拒绝，携带 `reason` 与 `confidence` |
| `done` | 流结束 |
| `error` | 生成失败，携带 `detail` |

### curl 示例

非流式：

```bash
curl -X POST "http://localhost:8000/api/v1/answer/doc_xxx?stream=false" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是集合？", "top_k": 5}'
```

流式（`-N` 关闭缓冲）：

```bash
curl -N -X POST "http://localhost:8000/api/v1/answer/doc_xxx?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是集合？", "top_k": 5}'
```

低置信被拒：

```bash
curl -X POST "http://localhost:8000/api/v1/answer/doc_xxx" \
  -H "Content-Type: application/json" \
  -d '{"query": "量子纠缠是什么？"}'
```

### 引用约定

答案中的 `[n]` 对应 `sources` 数组中 `index=n` 的条目；调用方按编号映射到
`page_num` 即可定位教材页码。

## 分组联合索引

分组是共享检索空间：组内每个文档的块写入同一个 Chroma 集合（元数据带
`doc_id/filename`），检索与问答跨全组一次完成；新文档上传后通过
`POST /api/v1/groups/{id}/index/{doc_id}` 增量编入，无需重建整组。

```bash
# 1. 创建分组
curl -X POST http://localhost:8000/api/v1/groups \
  -H "Content-Type: application/json" -d '{"name": "必修一"}'

# 2. 把存量已索引文档迁移进分组（后台任务，基于 KG 数据，不重新 OCR）
curl -X POST http://localhost:8000/api/v1/groups/<group_id>/migrate

# 3. 分组内联合检索 / 问答
curl -X POST http://localhost:8000/api/v1/groups/<group_id>/search \
  -H "Content-Type: application/json" -d '{"query": "什么是集合？", "top_k": 5}'
curl -X POST "http://localhost:8000/api/v1/groups/<group_id>/answer?stream=true" \
  -H "Content-Type: application/json" -d '{"query": "什么是集合？"}'
```

说明：文档的正式索引在分组模式下只存在于组集合（迁移或编入后旧单文档集合会被清理）；
移出分组或删除分组会把文档标记为 `pending`，需要重新索引。分组注册表持久化在
`groups.json`（默认 `GROUPS_FILE` 可配置）。

## 检索评测

```bash
# 对指定文档跑 30 题评测（改写开启），并生成 10 题人工复核文件
python scripts/evaluate_rag.py \
  --questions scripts/eval_questions.json \
  --doc-id <doc_id> --label rewrite \
  --output eval_report_rewrite.json --sample-human 10

# 对比两组报告（如 rewrite vs no_rewrite）
python scripts/compare_eval.py \
  --a eval_report_rewrite.json --b eval_report_no_rewrite.json \
  --output eval_report.json

# 从现有 KG 重建子块索引（不触发 OCR，仅调用嵌入 API）
python scripts/rebuild_index.py --doc-id <doc_id>
```

评测报告文件（`eval_report*.json`、`eval_human_review*.json`、`eval_results*.json`）
已加入 `.gitignore`，不提交仓库。

## 环境变量

完整列表见 [.env.example](.env.example)。除基础项外，常用检索配置：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `RETRIEVAL_CANDIDATE_K` | 20 | 每路召回候选深度（RRF 后再取 top_k） |
| `BM25_ENABLED` | true | BM25 稀疏检索路开关 |
| `CHAPTER_QUERY_ROUTING` | true | 章节类查询路由 |
| `ANSWER_CONFIDENCE_THRESHOLD` | 0.55 | 低置信度拒绝阈值（0 关闭） |
| `LLM_RERANK_ENABLED` | true | 概念类查询 LLM 重排开关 |
| `LLM_RERANK_TOP_N` | 20 | 重排候选池大小 |
| `ANSWER_TEMPERATURE` | 0.3 | 答案生成温度 |
| `ANSWER_MAX_TOKENS` | 1024 | 答案生成最大 token 数 |
| `SUBCHUNK_ENABLED` | true | 子块化索引开关 |
| `SUBCHUNK_MAX_TOKENS` | 160 | 子块 token 上限 |
| `GROUPS_FILE` | ./groups.json | 分组注册表路径 |

兼容旧的拼写错误键名（如 `QWEN3_API_KEY`、`PADLLEOCR_API_KEY`、
`QWEM3_MODLE_NAME`），推荐迁移到新命名。

## 测试与代码质量

```bash
uv run pytest               # 运行测试
uv run ruff check .         # 代码检查
uv run ruff format .        # 代码格式化
```

## 已知限制

- 答案生成为单轮问答，暂无多轮对话历史；生成质量依赖检索片段质量。
- 章节/栏目类查询（如章末复习栏目）召回仍有短板；章节类答案常落在 top2-4。
- 索引任务为进程内 `BackgroundTasks`，服务重启会丢失进行中的任务。
- `src/dots-ocr/` 为早期原型脚本，已由 `src/core/parser.py` 替代，确认无用后可删除。
- 生产环境建议将任务队列替换为 Celery / RQ 等持久化队列，
  并增加鉴权、限流与反向代理 + HTTPS。
