"""
配置管理模块

使用 pydantic-settings 从 .env 文件读取所有配置项。
其他模块通过 get_settings() 获取配置单例，避免重复读取。

⚠️ 配置优先级（踩过坑，务必记住）：
    真实环境变量  >  .env 文件  >  代码里的默认值

也就是说，只要系统里存在同名环境变量，`.env` 里写的值就会被**静默忽略**。
`OPENAI_API_KEY` 是重灾区——很多 AI 工具会把它设成系统级（Machine scope）变量。
因此本项目的 LLM 密钥优先读 `PAPERLENS_API_KEY`，`OPENAI_API_KEY` 仅作兼容回退。
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置，所有值从 .env 文件读取"""

    # ---------- LLM 配置 ----------
    # 变量名用 PAPERLENS_API_KEY 而不是 OPENAI_API_KEY，是被真实事故逼出来的：
    # 机器上装过的某个 AI 工具把 OPENAI_API_KEY 设成了 Machine 级环境变量，
    # 它的优先级高于 .env，导致"明明改了 .env，还是 401 Invalid token"。
    # AliasChoices 让 PAPERLENS_API_KEY 优先命中，同时保留旧名兼容已有配置。
    openai_api_key: str = Field(
        default="sk-your-key",
        validation_alias=AliasChoices("PAPERLENS_API_KEY", "OPENAI_API_KEY"),
    )
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model_name: str = "gpt-3.5-turbo"
    # 论文问答偏事实型，温度调低可减少"自由发挥"；对比/综述类可适当提高到 0.5
    llm_temperature: float = 0.3
    embedding_model_name: str = "text-embedding-ada-002"

    # ---------- 嵌入模型提供方 ----------
    # openai      : 走 OPENAI_BASE_URL 的 /embeddings 端点（要求该网关支持嵌入）
    # huggingface : 本地推理（sentence-transformers），不消耗额度，首次需下载模型
    embedding_provider: str = "openai"

    # ---------- MySQL 配置 ----------
    mysql_host: str = "mysql"           # Docker默认，本地开发改为 localhost
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "kb_copilot_2026"
    mysql_database: str = "kb_copilot"

    # ---------- FastAPI 配置 ----------
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    log_level: str = "INFO"
    log_dir: str = "./logs"

    # ---------- 文档处理配置 ----------
    chunk_size: int = 500
    chunk_overlap: int = 50
    retriever_top_k: int = 4
    # 入库前过滤参考文献/致谢/页眉页脚等噪声块（见 utils/content_filter.py）
    filter_references: bool = True
    # 查询时按问题意图给章节加权（见 utils/section_detector.py）
    section_aware_retrieval: bool = True
    # 单文件上传大小上限（MB）。0 表示不限制，仅建议在可信环境这么做。
    max_upload_mb: int = 50

    # ---------- 重排（cross-encoder） ----------
    # 开启后流程：向量召回 rerank_candidates 条 → cross-encoder 逐对打分 → 取 top_k
    # 关闭时退化为纯向量检索（与旧行为一致）
    rerank_enabled: bool = True
    rerank_model: str = "BAAI/bge-reranker-base"
    # 参与重排的候选数：太小重排没意义，太大 CPU 上会变慢
    rerank_candidates: int = 30

    # ---------- 跨文献对比 ----------
    # 一次对比最多允许选几篇：太多会撑爆上下文，结论也会糊成一团
    compare_max_documents: int = 4
    # 每篇论文各取多少个文本块参与对比
    compare_chunks_per_doc: int = 4
    # 对比上下文的字符预算（粗按 1 汉字 ≈ 1 字符估），超出后按文档数均分再截断
    compare_max_context_chars: int = 12000
    # 未指定对比维度时默认对齐的章节，对应 utils/section_detector.py 的 section key
    compare_default_section: str = "method"

    # ---------- 文件存储路径 ----------
    upload_dir: str = "./data/uploads"
    vector_store_dir: str = "./data/vector_store"

    @property
    def max_upload_bytes(self) -> int:
        """上传大小上限（字节），0 表示不限制"""
        return self.max_upload_mb * 1024 * 1024

    @property
    def mysql_url(self) -> str:
        """构建 MySQL 连接字符串（SQLAlchemy 格式）"""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            f"?charset=utf8mb4"
        )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,      # 环境变量名不区分大小写
        extra="ignore",            # 忽略 .env 中多余的变量
    )


@lru_cache()
def get_settings() -> Settings:
    """
    获取配置单例。

    使用 lru_cache 保证整个应用生命周期内只读取一次 .env 文件。
    如果需要刷新配置，调用 get_settings.cache_clear()。
    """
    return Settings()
