# 📚 论文阅读助手 PaperLens

> 面向科研文献的 RAG 问答与**跨文献对比**系统。
> 上传论文 → 自动解析、切分、清洗、标注章节 → 提问获得带页码/章节的引用回答 → 多篇论文按维度生成对比表。

---

## 📖 它解决什么问题

读一篇论文，你要在摘要、方法、实验、结论之间来回翻；读几篇同方向的论文，还得在心里维护一张对照表。这个助手把这两件事变成能直接问的问题。

| 你想知道 | 它怎么答 |
|---|---|
| 这篇的创新点是什么 | 对齐摘要 + 引言。这两章在全文只占 3–4 块，按朴素 top-k 很容易被方法章里同样高频的「提出 / 框架」挤掉 |
| 这个方法具体怎么做 | 对齐方法章，按原文顺序列出步骤 |
| 实验结果和对比基线 | 对齐实验章，指标、数据集名照抄原文，不换算、不约等于 |
| 这几篇论文差在哪 | 每篇独立检索、按同一维度对齐，输出对照表，每个单元格带 `[A-2]` 形式的出处 |
| 它到底没写什么 | 片段不足时直接说「不足以回答」，并指出缺哪一章，不推测、不补背景知识 |

论文这种文档有几个特性会让检索失准，所以链路里多了两道处理：

- **参考文献会抢名额**。论文尾部的参考文献全是领域术语，向量上与问题高度相似，实测一篇 10 页论文的 top-8 里能占到 1–2 条，因此在向量化之前先按块过滤。
- **检索顺序不是阅读顺序**。对比「方法流程」时，如果模型看到的是第 12 段 → 第 3 段 → 第 8 段，会把步骤读反，所以篇内要按块序号还原。

其余部分（切分、向量化、重排、引用溯源）走标准 RAG 链路。

---

## 🔄 文档处理流水线

```
上传文件
  → 解析        PyPDFLoader（PDF，逐页保留页码）/ TextLoader（TXT/MD，UTF-8 失败回退 GBK）
  → 切分        RecursiveCharacterTextSplitter（chunk_size / chunk_overlap 可配）
  → 内容清洗    丢弃参考文献 / 致谢 / 过短残片（见 utils/content_filter.py）
  → 章节标注    按标题正则给每块打 abstract/introduction/method… 标签
  → 向量化      FAISS，metadata 携带 document_id / chunk_index / page_number / section
  → 入库        文本块写 MySQL，文档状态置为 completed
```

两个容易忽略的细节：

- **清洗必须在向量化之前**。论文尾部的参考文献里全是领域术语，入库后一定会污染检索。
- **页码全链路统一为 1 起**。`PyPDFLoader` 给的是 0 起，在 `to_display_page()` 里统一转换，之后数据库、FAISS 元数据、引用卡片、送进模型的上下文用的都是同一个数字。

---

## 🧩 关键设计

### 章节感知检索

**入库时**：按中英文标题正则给每个文本块打章节标签（7 类）：

```
abstract / introduction / related_work / method / experiment / conclusion / other
```

标题只在它所在的那个块里出现，后续块靠「继承上一个状态」获得章节，所以必须按文档顺序遍历。识别用了两级正则：严格式要求整行只有「编号 + 标题词」，宽松式允许标题后紧跟分隔符（`Abstract—We propose…` 这种排版），但只对短行启用，避免把 `Results show that…` 这类正文句子当成标题。

**查询时**：从问题里推断用户想看哪几章，再对命中章节**加权排序**：

| 用户问 | 命中章节 |
|--------|---------|
| 「创新点 / idea / 贡献 / 解决什么问题」 | 摘要 + 引言 |
| 「怎么实现 / 架构 / 流程 / 训练」 | 方法 |
| 「实验 / 对比 / 消融 / 数据集」 | 实验 |
| 「局限 / 总结 / 未来工作」 | 结论 |

**加权而非硬过滤**：硬过滤的写法更简单，但章节缺失时会直接返回空结果。论文排版差异很大，没有 Introduction 标题的情况不少见；加权在章节缺失时只是退化成普通检索，不会让整个问答失败。

另外，加权只影响**排序**，不改变展示给用户的相关度分数（`rank_score` 与 `base_score` 分开），避免出现一个虚高的 0.98。

### 跨文献对比

它不能复用单篇问答的检索路径，原因见下面第 1 条。流程：

```
选中 N 篇论文 + 对比维度
  → 维度词推导目标章节（"实验" → experiment 章）
  → 每篇论文 **独立** 检索：向量召回（**不走 cross-encoder 重排**）→ 章节对齐
  → 篇内按 chunk_index 升序重排        ← 还原阅读顺序
  → 按字符预算均分后截断              ← 防止某篇独大
  → 拼成【论文A】/【论文B】分段上下文，带 [A-2] 式编号
  → 模型输出对比表 + 关键差异
```

五处设计要点：

1. **每篇独立配额**，不是全库 top-k。否则长论文会把其他论文挤空，对比表里全是「未提及」。
2. **章节对齐带回退**。对齐后不足 2 块时退回全部候选。宁可章节不齐，也不要整篇缺席。
3. **篇内按 `chunk_index` 重排**。检索返回的是相关度顺序，对比需要阅读顺序；少了这一步，模型会把方法步骤读反。单篇问答不受影响，因为它只关心片段内容，不关心先后。
4. **预算按篇数均分**。给某篇多、给某篇少，模型会对某篇「看得更细」，对比结论就带上偏差。单块超预算时截断而非整块丢弃。
5. **对比刻意不做 cross-encoder 重排**。重排要按篇数重复 N 次，成本直接乘上去
   （实测 CPU 上单篇重排 30 条候选约 10s）；而对比的准确性主要由「章节对齐」保证，
   候选池又被章节过滤缩小，重排的边际收益抵不过延迟。需要时把 `ask()` 里那段
   重排逻辑搬进 `_compare_chunks_for_doc()` 即可。

上下文里的引用编号用 `[A-2]`（论文 A 的第 2 段），比全局流水号更适合对比场景，能同时表达「哪一篇的第几段」。前端会把这个编号和底部片段卡片一一对应，可以回查页码。

另外两个兜底：全部论文都没检到片段时**不调模型**（浪费钱且必然胡编），直接返回提示；单篇检索为空时在上下文里显式写「该论文没有检索到可用片段」。

---

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 📤 文档上传 | PDF / TXT / MD，大小与类型校验，文件名清洗防路径穿越 |
| 🧹 内容清洗 | 自动丢弃参考文献 / 致谢 / 过短残片（可关），丢弃超 40% 会告警防误伤 |
| 🏷️ 章节标注 | 入库时自动识别论文 7 类章节 |
| 🔍 两阶段检索 | 向量召回 30 条候选 → 章节加权 → bge-reranker 精排 → top-k |
| 📎 引用溯源 | 每条结论标注来源论文、页码（1 起）、章节、向量分/重排分 |
| 🔬 跨文献对比 | 多篇论文按维度生成对比表，带 `[A-2]` 式可回溯引用 |
| 💬 会话管理 | 多会话、可切换删除（软删）、URL 记录当前会话（刷新不丢） |
| 📋 文档管理 | 列表、状态、统计、删除（尽力清理向量索引，失败可 rebuild 修复） |
| 🔧 索引重建 | 换嵌入模型后不必重传文件，从 MySQL 文本块重建（含章节重推断） |
| 🐳 一键部署 | Docker Compose 起 mysql + backend + frontend |

---

## 🛠️ 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 前端 | **Streamlit** | 问答、对比、文档管理（演示用，非生产级交互） |
| 后端 | **FastAPI** | REST API、依赖注入、线程池承载阻塞调用 |
| AI 框架 | **LangChain** | 文档加载、切分、LCEL Chain |
| 大模型 | **OpenAI 兼容接口** | 回答与对比生成（DeepSeek / 中转网关均可） |
| 向量化 | **OpenAI Embeddings 或本地 sentence-transformers** | 由 `EMBEDDING_PROVIDER` 切换 |
| 重排 | **bge-reranker-base（cross-encoder）** | 本地 CPU 推理，懒加载，失败自动降级 |
| 向量库 | **FAISS** | 向量存储与检索（进程内，无独立服务） |
| 数据库 | **MySQL 8.0** | 文档元数据、文本块、会话、消息 |
| 部署 | **Docker + Compose** | 三服务编排 |

---

## 📁 项目结构

```
paperlens/
│
├── backend/
│   ├── main.py                 # 入口：注册路由、CORS、生命周期、健康检查
│   ├── config.py               # pydantic-settings 读 .env，lru_cache 单例
│   ├── database.py             # SQLAlchemy 连接池、连接重试、get_db 依赖
│   ├── models.py               # ORM: documents / document_chunks / chat_sessions / chat_messages
│   ├── schemas.py              # Pydantic 请求校验与响应序列化
│   │
│   ├── routers/
│   │   ├── documents.py        # 上传 / 列表 / 详情 / 删除 / 统计 / 重建索引
│   │   └── chat.py             # 论文问答 / 跨文献对比 / 会话与消息
│   │
│   ├── services/
│   │   ├── document_service.py # 流水线：解析 → 切分 → 清洗 → 向量化 → 入库
│   │   ├── vector_service.py   # FAISS 单例、线程安全写锁、删除时修复位置映射
│   │   ├── rerank_service.py   # cross-encoder 懒加载 + 失败降级
│   │   └── chat_service.py     # ask() 单篇问答 / compare() 跨文献对比
│   │
│   └── utils/
│       ├── section_detector.py # 章节识别 + 问题→章节意图映射（核心）
│       ├── content_filter.py   # 参考文献 / 致谢 / 过短块过滤
│       ├── file_parser.py      # PDF(PyPDFLoader) / TXT / MD 解析
│       └── logger.py           # loguru：控制台彩色 + 10MB 轮转，保留 7 天
│
├── frontend/
│   ├── app.py                  # 首页：状态面板、统计、导航
│   ├── utils.py                # 后端 API 封装（统一错误处理）
│   └── pages/
│       ├── 1_💬_Chat.py        # 问答：会话管理、引用展示
│       ├── 2_📄_Documents.py   # 文档：上传、列表、删除、重建索引
│       └── 3_🔬_文献对比.py    # 对比：多选论文 + 维度 + 对比表
│
├── scripts/init_db.sql         # 建库建表（4 张表，含索引与外键）
├── data/                       # 运行时数据（Docker volume）
│   ├── uploads/                # 原始论文
│   └── vector_store/           # FAISS 索引（index.faiss + index.pkl）
├── logs/                       # 日志文件（按天分文件 + 10MB 轮转）
├── .env.example
├── requirements.txt
├── Dockerfile                  # 后端与前端共用一个镜像，靠 compose 的 command 区分
├── docker-compose.yml
└── README.md
```

---

## ⚙️ 环境变量

复制 `.env.example` 为 `.env` 后按需修改。完整清单以 `backend/config.py` 为准。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PAPERLENS_API_KEY` | — | **必填**。兼容 OpenAI 协议的密钥。旧名 `OPENAI_API_KEY` 仍兼容，但优先级更低（见下方说明） |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | API 地址，官方/DeepSeek/中转网关均可 |
| `LLM_MODEL_NAME` | `gpt-3.5-turbo` | 聊天模型 |
| `LLM_TEMPERATURE` | `0.3` | 论文问答偏事实型，调低减少发挥 |
| `EMBEDDING_PROVIDER` | `openai` | `openai` 走网关 `/embeddings`；`huggingface` 本地推理 |
| `EMBEDDING_MODEL_NAME` | `text-embedding-ada-002` | 嵌入模型；HF 时填 `org/name` |
| `MYSQL_HOST` | `mysql` | Docker 用 `mysql`，本地开发改 `localhost` |
| `MYSQL_PORT` | `3306` | |
| `MYSQL_USER` | `root` | |
| `MYSQL_PASSWORD` | `kb_copilot_2026` | |
| `MYSQL_DATABASE` | `kb_copilot` | |
| `BACKEND_HOST` | `0.0.0.0` | |
| `BACKEND_PORT` | `8000` | |
| `LOG_LEVEL` | `INFO` | DEBUG / INFO / WARNING / ERROR |
| `LOG_DIR` | `./logs` | 日志目录 |
| `CHUNK_SIZE` | `500` | 切分大小（字符）。bge 类 512 token 上限，中文建议 ≤450 |
| `CHUNK_OVERLAP` | `50` | 切分重叠 |
| `RETRIEVER_TOP_K` | `4` | 单次问答检索块数（前端滑块会覆盖它） |
| `FILTER_REFERENCES` | `true` | 入库前过滤参考文献/致谢/过短块 |
| `SECTION_AWARE_RETRIEVAL` | `true` | 章节感知加权开关 |
| `MAX_UPLOAD_MB` | `50` | 单文件上限，`0` 为不限 |
| `RERANK_ENABLED` | `true` | 关闭后退化为纯向量检索 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | cross-encoder 模型 |
| `RERANK_CANDIDATES` | `30` | 参与重排的候选数（也是"召回 30 条"的来源） |
| `COMPARE_MAX_DOCUMENTS` | `4` | 一次最多对比几篇 |
| `COMPARE_CHUNKS_PER_DOC` | `4` | 每篇取几段 |
| `COMPARE_MAX_CONTEXT_CHARS` | `12000` | 对比上下文总预算，按篇数均分 |
| `COMPARE_DEFAULT_SECTION` | `method` | 未指定维度时对齐的章节 |
| `UPLOAD_DIR` | `./data/uploads` | 原始文件落盘目录 |
| `VECTOR_STORE_DIR` | `./data/vector_store` | FAISS 索引目录 |

> 📌 **另有三个变量不在 `config.py` 里**：`HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`、`HF_ENDPOINT`。
> 它们由 `huggingface_hub` / `transformers` 直接读取**进程环境**，且只在模块导入时求值一次。
> `.env` 里的值由 `backend/__init__.py` 的 `load_dotenv()` 注入（必须早于模型库被导入），
> 所以照常写在 `.env` 里即可。国内网络建议常开 `HF_HUB_OFFLINE=1`。

> ⚠️ **配置优先级：真实环境变量 > `.env` > 代码默认值。**
> 只要系统里存在同名环境变量，`.env` 里的值就会被**静默忽略**：不报错、不警告，只是不生效。
> 这正是密钥变量叫 `PAPERLENS_API_KEY` 而不是 `OPENAI_API_KEY` 的原因：后者太通用，
> 很容易被机器上别的 AI 工具设成系统级变量，导致"改了 `.env` 却一直 401"。
> `config.py` 用 `AliasChoices("PAPERLENS_API_KEY", "OPENAI_API_KEY")` 保证前者优先。

---

## 🚀 快速启动

### 方式一：Docker（推荐）

```bash
cp .env.example .env      # Windows: copy .env.example .env
# 编辑 .env，填 PAPERLENS_API_KEY
docker-compose up --build # 首次构建约 3–5 分钟
```

- 前端：http://localhost:8501
- API 文档：http://localhost:8000/docs

MySQL 首次启动会自动执行 `scripts/init_db.sql` 建库建表；`depends_on` 配置了健康检查，后端会等数据库就绪再启动。

```bash
docker-compose up -d --build   # 后台运行
docker-compose logs -f backend # 查看日志
docker-compose down            # 停止
docker-compose down -v         # 停止并清空数据卷
```

### 方式二：本地开发

前置：**Python 3.11**、MySQL 8.0

> ⚠️ **必须是 3.11 或 3.12，3.13+ 装不上。**
> `requirements.txt` 里锁定的 `faiss-cpu==1.8.0.post1` 没有 cp313/cp314 的 wheel，
> 用 3.13/3.14 会直接报 `No matching distribution found for faiss-cpu==1.8.0.post1`。
>
> **不要靠"升 faiss 版本"绕过**：新版 langchain 移除了 `langchain.text_splitter`
> 这个导入路径，而 `document_service.py` 正是从那里导入的，升依赖会让代码直接报错。

#### 方式 A：venv（本机 `python` 已是 3.11 时）

```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

#### 方式 B：conda（本机默认 python 是 3.13/3.14 时推荐）

```bash
conda create -n paperlens python=3.11 -y
conda activate paperlens
python --version                 # 必须显示 3.11.x
pip install -r requirements.txt
```

#### 通用步骤

```bash
cp .env.example .env             # 改 MYSQL_HOST=localhost，填 PAPERLENS_API_KEY
mysql -u root -p < scripts/init_db.sql

# 终端 1：后端
python -m backend.main

# 终端 2：前端
cd frontend && streamlit run app.py
```

装完依赖先自检：

```bash
python -c "import faiss, langchain, fastapi, streamlit, sqlalchemy, pymysql; print('deps ok')"
```

---

## 📡 API

启动后访问 Swagger UI：http://localhost:8000/docs

**文档管理**（前缀 `/api/v1/documents`）

| 接口 | 方法 | 说明 |
|------|------|------|
| `/upload` | POST | 上传并同步处理（解析→切分→清洗→向量化→入库） |
| `` （即 `/api/v1/documents`） | GET | 论文列表（分页，默认 10 条/页，上限 100） |
| `/stats/overview` | GET | 论文库统计 |
| `/rebuild-index` | POST | 用 MySQL 文本块重建 FAISS 索引 |
| `/{document_id}` | GET | 论文详情 |
| `/{document_id}` | DELETE | 删除论文（数据库 + 本地文件 + 尽力清理向量） |

**论文问答**（前缀 `/api/v1/chat`）

| 接口 | 方法 | 说明 |
|------|------|------|
| `` （即 `/api/v1/chat`） | POST | 单篇问答 |
| `/compare` | POST | **跨文献对比** |
| `/sessions` | GET | 会话列表（按最后活跃时间倒序） |
| `/sessions/{id}/messages` | GET | 会话消息历史 |
| `/sessions/{id}` | DELETE | 删除会话（软删） |

**系统**

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查（数据库 + 向量库状态） |
| `/` | GET | 根路径，返回服务信息 |

> 共 **11 条路由**（`/{document_id}` 同时支持 GET 与 DELETE，合计 12 个接口方法）+ 根路径。注意 `/stats/overview` 与 `/rebuild-index` 必须声明在 `/{document_id}` 之前，否则会被当成文档 ID 解析。

**对比接口示例：**

```bash
curl -X POST http://localhost:8000/api/v1/chat/compare \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": [1, 2],
    "aspect": "实验设置与评测结果",
    "chunks_per_doc": 4
  }'
```

返回 `answer`（对比表 + 关键差异）、`groups`（每篇的引用片段，含 `ref` 编号）、`stats`（片段数 / 上下文字符数 / 是否被截断 / 耗时）。

---

## 🗄️ 数据库

4 张表，关系：`documents 1—N document_chunks`、`chat_sessions 1—N chat_messages`。

| 表 | 说明 | 核心字段 |
|----|------|---------|
| `documents` | 论文元数据 | filename, file_type, file_size, file_path, status, chunk_count |
| `document_chunks` | 文本块 | document_id, chunk_index, content, page_number, vector_id |
| `chat_sessions` | 会话 | title, is_active（软删标记）, message_count |
| `chat_messages` | 消息 | session_id, question, answer, source_chunks(JSON), response_time |

⚠️ **`document_chunks` 没有 section 字段**，章节标签只存在于 FAISS 元数据里。所以 `/rebuild-index` 会按块顺序**重新推断**章节（见 `document_service.rebuild_index`），而不是从库里读。

对比结果同样写进 `chat_messages`，历史里可以回看。

---

## 🔮 下一步

- [ ] **检索评测集**：50–100 条论文问题（事实型 / 跨段落 / 全局型 / 无答案 / 对比型），
      统计 Recall@5、答案准确率、引用准确率、拒答正确率；用数据决定
      `CHUNK_SIZE`、章节加权强度、`RERANK_CANDIDATES` 的取值
- [ ] 多轮上下文：最近 N 轮拼进 Prompt，带 token 预算
- [ ] 对比模式：单篇摘要 → 摘要级对比（降低长上下文压力）
- [ ] 上传改异步：任务队列 + 进度查询，避免大论文阻塞请求
- [ ] 表格/公式抽取
- [ ] 流式输出（SSE）
- [ ] 换 pgvector / Milvus 支持多副本

---

## 📄 License

MIT，见 [LICENSE](LICENSE)。
