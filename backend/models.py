"""
ORM 数据模型

定义四张核心表：
- Document:       文档元数据（文件名、类型、大小、状态等）
- DocumentChunk:  文档切分后的文本块
- ChatSession:    聊天会话（一次对话的容器）
- ChatMessage:    聊天消息（一问一答为一条记录）

设计说明：
    将原来的单表 ChatHistory 拆分为 ChatSession + ChatMessage，
    这样可以支持多轮对话分组、会话重命名、独立删除等功能。
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float,
    DateTime, JSON, ForeignKey, Index, Boolean,
)
from sqlalchemy.orm import relationship

from backend.database import Base


class Document(Base):
    """
    文档元数据表

    记录用户上传的每一个文档的基本信息和处理状态。
    一个文档可以被切分为多个 DocumentChunk。
    """
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    filename = Column(String(255), nullable=False, comment="原始文件名")
    file_type = Column(String(10), nullable=False, comment="文件类型: pdf/txt/md")
    file_size = Column(Integer, nullable=False, comment="文件大小(字节)")
    file_path = Column(String(500), nullable=False, comment="服务器存储路径")
    chunk_count = Column(Integer, default=0, comment="切分后的文本块数量")
    status = Column(
        String(20), default="processing",
        comment="处理状态: processing(处理中) / completed(已完成) / failed(失败)"
    )
    error_message = Column(String(500), nullable=True, comment="处理失败时的错误信息")
    created_at = Column(DateTime, default=datetime.utcnow, comment="上传时间")
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        comment="最后更新时间"
    )

    # 关联关系：一个文档 → 多个文本块（级联删除）
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="dynamic",  # 大量切片时延迟加载，避免一次性加载全部
    )

    __table_args__ = (
        Index("idx_doc_status", "status"),
        Index("idx_doc_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, filename='{self.filename}', status='{self.status}')>"


class DocumentChunk(Base):
    """
    文档文本块表

    一个文档被切分后得到的文本块，每个块对应 FAISS 中的一个向量。
    page_number 仅对 PDF 文件有效。
    """
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    document_id = Column(
        Integer,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属文档 ID"
    )
    chunk_index = Column(Integer, nullable=False, comment="块在文档中的序号(从 0 开始)")
    content = Column(Text, nullable=False, comment="文本块内容")
    page_number = Column(Integer, nullable=True, comment="所在页码(1 起，仅 PDF 有效)")
    vector_id = Column(String(100), nullable=True, comment="在 FAISS 向量库中的 ID")
    created_at = Column(DateTime, default=datetime.utcnow, comment="入库时间")

    # 关联关系：多个文本块 → 一个文档
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("idx_chunk_document_id", "document_id"),
        Index("idx_chunk_vector_id", "vector_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, "
            f"index={self.chunk_index})>"
        )


class ChatSession(Base):
    """
    聊天会话表

    一个会话代表一次完整的对话，包含多条消息。
    用户可以创建多个会话，每个会话有独立的标题。
    """
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    title = Column(
        String(255), default="新对话",
        comment="会话标题(默认取第一个问题的前 50 字)"
    )
    is_active = Column(Boolean, default=True, comment="是否激活(软删除标记)")
    message_count = Column(Integer, default=0, comment="消息数量")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
        comment="最后活跃时间"
    )

    # 关联关系：一个会话 → 多条消息（级联删除）
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",  # 按时间排序
    )

    __table_args__ = (
        Index("idx_session_active", "is_active"),
        Index("idx_session_updated", "updated_at"),
    )

    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, title='{self.title}')>"


class ChatMessage(Base):
    """
    聊天消息表

    一条记录 = 一问一答（用户问题 + AI 回答）。
    source_chunks 以 JSON 格式存储引用来源，结构示例：
    [
        {
            "document_name": "attention-is-all-you-need.pdf",
            "chunk_content": "We employ a residual connection around each sub-layer...",
            "page": 3,
            "section": "method",
            "relevance_score": 0.92,
            "ref": "A-2"
        }
    ]
    """
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="自增主键")
    session_id = Column(
        Integer,
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        comment="所属会话 ID"
    )
    question = Column(Text, nullable=False, comment="用户提问")
    answer = Column(Text, nullable=False, comment="AI 回答")
    source_chunks = Column(JSON, nullable=True, comment="引用来源(JSON 数组)")
    response_time = Column(Float, nullable=True, comment="AI 响应耗时(秒)")
    created_at = Column(DateTime, default=datetime.utcnow, comment="消息时间")

    # 关联关系：多条消息 → 一个会话
    session = relationship("ChatSession", back_populates="messages")

    __table_args__ = (
        Index("idx_msg_session_id", "session_id"),
        Index("idx_msg_created_at", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<ChatMessage(id={self.id}, session_id={self.session_id}, "
            f"question='{self.question[:30]}...')>"
        )
