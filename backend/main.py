"""
FastAPI 应用入口

职责：
- 创建 FastAPI 应用实例
- 注册所有路由
- 配置 CORS 中间件
- 启动时初始化数据库和日志
- 提供健康检查接口
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.database import init_db, check_db_connection
from backend.services.vector_service import VectorService
from backend.utils.logger import logger, setup_logger
from backend.routers import documents, chat
from backend.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理。

    startup: 初始化日志、创建目录、连接数据库、同步表结构
    shutdown: 清理资源（当前无需特殊处理）
    """
    # ---- 启动阶段 ----
    setup_logger()
    logger.info("=" * 60)
    logger.info("🚀 论文阅读助手 后端启动中...")

    settings = get_settings()

    # 确保必要目录存在（全部来自配置，避免 cwd 不同导致写到别处）
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.vector_store_dir, exist_ok=True)
    os.makedirs(settings.log_dir, exist_ok=True)
    logger.info("📁 数据目录已就绪: uploads={}, vector_store={}", settings.upload_dir, settings.vector_store_dir)

    # ---- 启动自检：把生效的模型配置打进日志 ----
    # 配置由 get_settings() 的 lru_cache 缓存，改了 .env 必须重启进程才生效，
    # 这行日志用来快速确认"进程到底在用哪套配置"（不打印完整密钥）。
    _api_key = settings.openai_api_key or ""
    _key_tail = _api_key[-4:] if len(_api_key) >= 4 else ""
    logger.info(
        "🔧 生效配置: LLM='{}', base_url='{}', embedding_provider='{}', "
        "embedding_model='{}', temperature={}, api_key=<len={}, ...{}>",
        settings.llm_model_name,
        settings.openai_base_url,
        settings.embedding_provider,
        settings.embedding_model_name,
        settings.llm_temperature,
        len(_api_key),
        _key_tail,
    )

    # 初始化数据库（含连接重试）
    try:
        init_db()
        logger.info("✅ 数据库初始化完成")
    except Exception as e:
        logger.error("❌ 数据库初始化失败: {}", str(e))
        raise

    # 预热向量服务：本地 embedding 模型首次加载要几秒到几十秒，放在启动时做，
    # 避免第一个上传/问答请求被这段时间拖到前端超时。
    # 失败不阻断启动——模型没下载或依赖没装时，接口层会给出明确报错。
    try:
        VectorService.get_instance()
        logger.info("✅ 向量服务已预热")
    except Exception as e:
        logger.warning("⚠️ 向量服务预热失败（首次使用时重试）: {}", str(e))

    logger.info("✅ 后端启动完成，监听 {}:{}", settings.backend_host, settings.backend_port)
    logger.info("📖 API 文档: http://localhost:{}/docs", settings.backend_port)
    logger.info("=" * 60)

    yield  # ---- 应用运行中 ----

    # ---- 关闭阶段 ----
    logger.info("后端服务正在关闭...")


# ============================================================
# 创建 FastAPI 应用
# ============================================================
app = FastAPI(
    title="论文阅读助手 API",
    description=(
        "面向科研文献的 RAG 问答与跨文献对比系统。\n\n"
        "- 单篇问答：章节感知检索 + 两阶段精排，回答带页码/章节引用\n"
        "- 跨文献对比：每篇独立检索配额，按维度对齐后生成对比表\n\n"
        "技术栈：FastAPI + LangChain + FAISS + MySQL"
    ),
    version="1.1.0",
    docs_url="/docs",       # Swagger UI 地址
    redoc_url="/redoc",     # ReDoc 地址
    lifespan=lifespan,
)

# ============================================================
# CORS 中间件
# ============================================================
# 前端是 Streamlit 服务端渲染，浏览器并不直接跨域调用本 API；这里放开只为
# 方便本地调试（Swagger UI、curl、自己写的小页面）。
# 注意：allow_credentials 不能与 allow_origins=["*"] 同时使用（浏览器会拒绝
# 带凭据的通配来源），本项目没有 Cookie 鉴权，所以关掉 credentials。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 注册路由
# ============================================================
app.include_router(documents.router)
app.include_router(chat.router)


# ============================================================
# 健康检查接口
# ============================================================
@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["系统"],
    summary="健康检查",
    description="检查后端服务、数据库连接、向量库状态。数据库异常时 status=degraded。"
)
def health_check():
    """
    健康检查。

    返回三个维度：
    - status:       healthy（依赖都正常）/ degraded（服务活着但数据库不可用）
    - database:     connected / disconnected
    - vector_store: loaded（内存中已有向量）/ on-disk（磁盘有索引但未加载）/ empty

    这里刻意不触发向量服务初始化：健康检查可能被前端每几秒调一次，
    不能因为一次探测就去加载几百 MB 的 embedding 模型。
    """
    settings = get_settings()

    db_ok = check_db_connection()

    vector_service = VectorService.peek_instance()
    if vector_service is not None:
        vector_status = "loaded" if vector_service.total_vectors > 0 else "empty"
    else:
        index_file = os.path.join(settings.vector_store_dir, "index.faiss")
        vector_status = "on-disk" if os.path.exists(index_file) else "empty"

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        database="connected" if db_ok else "disconnected",
        vector_store=vector_status,
    )


# ============================================================
# 根路径
# ============================================================
@app.get("/", tags=["系统"], summary="根路径")
def root():
    return {
        "name": "论文阅读助手 API",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# ============================================================
# 直接运行入口: python -m backend.main
# ============================================================
if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
