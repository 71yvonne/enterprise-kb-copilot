"""
聊天路由

提供论文问答、跨文献对比、会话管理、消息历史等接口。
问答接口调用 ChatService.ask 实现完整 RAG：检索 → 构建上下文 → LLM 生成 → 保存记录。
对比接口调用 ChatService.compare，走"每篇独立配额 + 章节对齐"的另一条检索策略。

关于同步/异步：
    问答链路包含 FAISS 检索（CPU）和 LLM HTTP 调用（阻塞），全部是同步代码。
    因此路由声明为 def 而非 async def，由 FastAPI 放入线程池执行，避免堵住
    事件循环——否则一次问答期间健康检查都会超时。
    对比接口同理，而且它的检索轮次是"论文数 × 1 次"，更容易阻塞。
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas import (
    BaseResponse,
    ChatRequest,
    ChatResponse,
    ChatResponseData,
    CompareGroup,
    CompareRequest,
    CompareResponse,
    CompareResponseData,
    CompareStats,
    MessageInfo,
    MessageListData,
    MessageListResponse,
    SessionInfo,
    SessionListData,
    SessionListResponse,
    SourceChunk,
)
from backend.services.chat_service import ChatService
from backend.utils.logger import logger

router = APIRouter(prefix="/api/v1/chat", tags=["论文问答"])


# ----------------------------------------------------------
# POST / — 单篇问答
# ----------------------------------------------------------
@router.post(
    "",
    response_model=ChatResponse,
    summary="论文问答",
    description="接收问题，从已上传论文中检索相关片段，生成回答并附带引用来源（文件名/页码/章节）。"
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):
    """
    论文问答完整流程（具体实现见 ChatService.ask）：
    1. 获取/创建聊天会话
    2. FAISS 向量检索 + 章节加权 + cross-encoder 精排
    3. 构建带 [来源i] 编号的上下文
    4. 调用 LLM 生成回答
    5. 保存消息记录到 MySQL
    6. 返回回答 + 引用来源 + 耗时
    """
    try:
        result = ChatService(db).ask(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
        )
    except ValueError as e:
        # 会话不存在属于客户端问题，映射成 404 而不是 500
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception("问答失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}") from e

    return ChatResponse(
        code=200,
        message="success",
        data=ChatResponseData(
            answer=result["answer"],
            sources=[
                SourceChunk(
                    document_name=s["document_name"],
                    chunk_content=s["chunk_content"],
                    page=s.get("page"),
                    section=s.get("section"),
                    relevance_score=s.get("relevance_score"),
                    rerank_score=s.get("rerank_score"),
                    chunk_index=s.get("chunk_index"),
                    ref=s.get("ref"),
                )
                for s in result["sources"]
            ],
            session_id=result["session_id"],
            message_id=result["message_id"],
            response_time=result.get("response_time"),
        ),
    )


# ----------------------------------------------------------
# POST /compare — 跨文献对比
# ----------------------------------------------------------
@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="跨文献对比",
    description=(
        "选中 2~N 篇论文，按指定维度（默认按方法章）对齐后生成结构化对比表。"
        "与普通问答的区别：每篇论文独立分配检索配额，避免长论文挤空短论文；"
        "篇内按块序号还原阅读顺序，避免模型把方法步骤读反。"
    ),
)
def compare(
    request: CompareRequest,
    db: Session = Depends(get_db),
):
    """
    跨文献对比（具体实现见 ChatService.compare）。

    参数不合法（论文不足 2 篇 / 超过上限 / 论文未处理完 / 没给对比依据）
    一律映射为 400，让前端能把原因直接展示出来。
    """
    try:
        result = ChatService(db).compare(
            document_ids=request.document_ids,
            question=request.question,
            aspect=request.aspect,
            chunks_per_doc=request.chunks_per_doc,
            session_id=request.session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("跨文献对比失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"对比处理失败: {str(e)}") from e

    return CompareResponse(
        code=200,
        message="success",
        data=CompareResponseData(
            answer=result["answer"],
            groups=[
                CompareGroup(
                    document_id=g["document_id"],
                    document_name=g["document_name"],
                    letter=g["letter"],
                    chunks=[SourceChunk(**chunk) for chunk in g["chunks"]],
                )
                for g in result["groups"]
            ],
            stats=CompareStats(**result["stats"]),
            session_id=result.get("session_id"),
            message_id=result.get("message_id"),
            response_time=result.get("response_time"),
        ),
    )


# ----------------------------------------------------------
# GET /sessions — 获取会话列表
# ----------------------------------------------------------
@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="获取会话列表",
    description="获取所有激活状态的聊天会话，按最后活跃时间倒序。"
)
def list_sessions(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数，最多 100"),
    db: Session = Depends(get_db),
):
    result = ChatService(db).list_sessions(page=page, page_size=page_size)

    return SessionListResponse(
        code=200,
        data=SessionListData(
            total=result["total"],
            sessions=[SessionInfo.model_validate(s) for s in result["sessions"]],
        ),
    )


# ----------------------------------------------------------
# GET /sessions/{session_id}/messages — 获取会话消息
# ----------------------------------------------------------
@router.get(
    "/sessions/{session_id}/messages",
    response_model=MessageListResponse,
    summary="获取会话消息",
    description="获取指定会话的所有聊天消息，按时间正序排列。"
)
def get_session_messages(
    session_id: int,
    db: Session = Depends(get_db),
):
    try:
        messages = ChatService(db).get_messages(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    return MessageListResponse(
        code=200,
        data=MessageListData(
            total=len(messages),
            session_id=session_id,
            messages=[
                MessageInfo(
                    id=msg.id,
                    session_id=msg.session_id,
                    question=msg.question,
                    answer=msg.answer,
                    sources=msg.source_chunks,
                    response_time=msg.response_time,
                    created_at=msg.created_at,
                )
                for msg in messages
            ],
        ),
    )


# ----------------------------------------------------------
# DELETE /sessions/{session_id} — 删除会话
# ----------------------------------------------------------
@router.delete(
    "/sessions/{session_id}",
    response_model=BaseResponse,
    summary="删除会话",
    description="软删除指定会话（标记为不活跃，消息保留）。"
)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
):
    if not ChatService(db).delete_session(session_id):
        raise HTTPException(status_code=404, detail="会话不存在")
    return BaseResponse(code=200, message="会话已删除")
