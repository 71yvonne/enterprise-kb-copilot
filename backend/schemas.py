"""
Pydantic 请求/响应模型

定义 API 的输入输出数据结构，用于：
- 请求参数自动校验
- 响应数据序列化
- Swagger 文档自动生成

命名约定：
- xxxRequest:  请求体
- xxxResponse: 完整响应（含 code + message + data）
- xxxInfo:     数据对象（嵌入到 Response.data 中）
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 通用响应模型
# ============================================================

class BaseResponse(BaseModel):
    """统一响应格式 — 所有 API 响应都遵循此结构"""
    code: int = Field(default=200, description="状态码，200 表示成功")
    message: str = Field(default="success", description="提示信息")


# ============================================================
# 文档相关模型
# ============================================================

class DocumentInfo(BaseModel):
    """单个文档的详细信息"""
    id: int = Field(description="文档 ID")
    filename: str = Field(description="原始文件名")
    file_type: str = Field(description="文件类型: pdf/txt/md")
    file_size: int = Field(description="文件大小(字节)")
    chunk_count: int = Field(description="切分的文本块数量")
    status: str = Field(description="处理状态: processing/completed/failed")
    error_message: Optional[str] = Field(default=None, description="失败原因")
    created_at: datetime = Field(description="上传时间")
    updated_at: Optional[datetime] = Field(default=None, description="更新时间")

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseResponse):
    """文档上传响应"""
    data: Optional[DocumentInfo] = None


class DocumentDetailResponse(BaseResponse):
    """单个文档详情响应"""
    data: Optional[DocumentInfo] = None


class DocumentListData(BaseModel):
    """文档列表数据"""
    total: int = Field(description="文档总数")
    page: int = Field(description="当前页码")
    page_size: int = Field(description="每页条数")
    documents: list[DocumentInfo] = Field(description="文档列表")


class DocumentListResponse(BaseResponse):
    """文档列表响应"""
    data: Optional[DocumentListData] = None


class DocumentStatsData(BaseModel):
    """论文库统计信息"""
    total_documents: int = Field(description="文档总数")
    total_chunks: int = Field(description="文本块总数")
    total_vectors: int = Field(description="FAISS 中的向量总数（可能包含已删除文档的陈旧向量）")
    completed: int = Field(description="处理完成的文档数")
    processing: int = Field(description="处理中的文档数")
    failed: int = Field(description="处理失败的文档数")


class DocumentStatsResponse(BaseResponse):
    """文档统计响应"""
    data: Optional[DocumentStatsData] = None


class RebuildIndexData(BaseModel):
    """索引重建结果"""
    vectors: int = Field(description="重建后的向量总数")
    documents: int = Field(description="参与重建的文档数")
    chunks: int = Field(description="参与重建的文本块数")


class RebuildIndexResponse(BaseResponse):
    """索引重建响应"""
    data: Optional[RebuildIndexData] = None


# ============================================================
# 聊天相关模型
# ============================================================

class ChatRequest(BaseModel):
    """聊天请求"""
    question: str = Field(
        ..., min_length=1, max_length=2000,
        description="用户问题"
    )
    session_id: Optional[int] = Field(
        default=None,
        description="会话 ID，为空则自动创建新会话"
    )
    top_k: Optional[int] = Field(
        default=None, ge=1, le=20,
        description="检索返回的文档块数量；不传则使用配置项 RETRIEVER_TOP_K"
    )


class SourceChunk(BaseModel):
    """引用来源片段"""
    document_name: str = Field(description="来源文档名")
    chunk_content: str = Field(description="引用的文本内容")
    page: Optional[int] = Field(default=None, description="所在页码(1 起)")
    section: Optional[str] = Field(
        default=None,
        description="所属章节: abstract/introduction/related_work/method/experiment/conclusion/other",
    )
    relevance_score: Optional[float] = Field(default=None, description="向量相似度分数")
    rerank_score: Optional[float] = Field(
        default=None,
        description="cross-encoder 重排分数(0~1)；未开启重排时为 None",
    )
    chunk_index: Optional[int] = Field(
        default=None, description="该片段在原文中的块序号，用于还原阅读顺序"
    )
    ref: Optional[str] = Field(
        default=None,
        description="对比模式下的引用编号，如 A-2 表示论文A的第2个片段；问答模式为 None",
    )


class ChatResponseData(BaseModel):
    """聊天响应数据"""
    answer: str = Field(description="AI 回答")
    sources: list[SourceChunk] = Field(default_factory=list, description="引用来源")
    session_id: int = Field(description="会话 ID")
    message_id: int = Field(description="消息 ID")
    response_time: Optional[float] = Field(default=None, description="本次回答耗时(秒)")


class ChatResponse(BaseResponse):
    """聊天响应"""
    data: Optional[ChatResponseData] = None


# ---------- 会话相关 ----------

class SessionInfo(BaseModel):
    """会话信息"""
    id: int = Field(description="会话 ID")
    title: str = Field(description="会话标题")
    message_count: int = Field(description="消息数量")
    is_active: bool = Field(description="是否激活")
    created_at: datetime = Field(description="创建时间")
    updated_at: Optional[datetime] = Field(default=None, description="最后活跃时间")

    model_config = {"from_attributes": True}


class SessionListData(BaseModel):
    """会话列表数据"""
    total: int
    sessions: list[SessionInfo]


class SessionListResponse(BaseResponse):
    """会话列表响应"""
    data: Optional[SessionListData] = None


# ---------- 消息相关 ----------

class MessageInfo(BaseModel):
    """单条消息信息"""
    id: int = Field(description="消息 ID")
    session_id: int = Field(description="会话 ID")
    question: str = Field(description="用户问题")
    answer: str = Field(description="AI 回答")
    sources: Optional[list[SourceChunk]] = Field(default=None, description="引用来源")
    response_time: Optional[float] = Field(default=None, description="响应耗时(秒)")
    created_at: datetime = Field(description="消息时间")

    model_config = {"from_attributes": True}


class MessageListData(BaseModel):
    """消息列表数据"""
    total: int
    session_id: int
    messages: list[MessageInfo]


class MessageListResponse(BaseResponse):
    """消息列表响应"""
    data: Optional[MessageListData] = None


# ============================================================
# 健康检查模型
# ============================================================

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="healthy", description="服务状态")
    database: str = Field(default="unknown", description="数据库连接状态")
    vector_store: str = Field(default="unknown", description="向量库状态")


# ============================================================
# 跨文献对比模型
# ============================================================

class CompareRequest(BaseModel):
    """跨文献对比请求"""
    document_ids: list[int] = Field(
        ..., min_length=2, max_length=8,
        description="参与对比的论文 ID，至少 2 篇（上限受 COMPARE_MAX_DOCUMENTS 约束）"
    )
    question: Optional[str] = Field(
        default=None, max_length=2000,
        description="对比问题；与 aspect 至少提供一个，提供后用于检索"
    )
    aspect: Optional[str] = Field(
        default=None, max_length=100,
        description="对比维度，如 方法 / 实验设置 / 创新点；后端据此推导要对齐的章节"
    )
    chunks_per_doc: Optional[int] = Field(
        default=None, ge=1, le=10,
        description="每篇论文取几段，不传则用配置 COMPARE_CHUNKS_PER_DOC"
    )
    session_id: Optional[int] = Field(
        default=None,
        description="会话 ID，为空则自动创建新会话"
    )


class CompareGroup(BaseModel):
    """单篇论文在本次对比中的片段分组"""
    document_id: int = Field(description="论文 ID")
    document_name: str = Field(description="论文文件名")
    letter: str = Field(description="本次对比中的编号：A/B/C…")
    chunks: list[SourceChunk] = Field(
        default_factory=list, description="该论文参与对比的片段（按原文顺序）"
    )


class CompareStats(BaseModel):
    """对比过程的运行指标 — 决定结论可不可信"""
    document_count: int = Field(description="参与对比的论文数")
    total_chunks: int = Field(description="实际送入上下文的片段总数")
    context_chars: int = Field(description="上下文总字符数")
    budget_per_doc: int = Field(default=0, description="每篇论文的字符预算")
    sections: list[str] = Field(default_factory=list, description="本次对齐的章节")
    truncated_documents: list[str] = Field(
        default_factory=list, description="因超预算被截断的论文（结论可能不完整）"
    )
    elapsed: Optional[float] = Field(default=None, description="总耗时(秒)")


class CompareResponseData(BaseModel):
    """跨文献对比响应数据"""
    answer: str = Field(description="对比结论（Markdown 表格 + 关键差异）")
    groups: list[CompareGroup] = Field(
        default_factory=list, description="每篇论文各自的引用片段"
    )
    stats: CompareStats = Field(description="运行指标")
    session_id: Optional[int] = Field(default=None, description="会话 ID；无结果时为 None")
    message_id: Optional[int] = Field(default=None, description="消息 ID；无结果时为 None")
    response_time: Optional[float] = Field(default=None, description="总耗时(秒)")


class CompareResponse(BaseResponse):
    """跨文献对比响应"""
    data: Optional[CompareResponseData] = None
