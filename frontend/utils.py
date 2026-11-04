"""
前端工具函数

封装对 FastAPI 后端的所有 HTTP 调用。
Streamlit 页面通过调用这些函数与后端通信，不直接使用 requests。

所有函数都包含异常处理，失败时返回统一格式的错误响应，
前端页面不需要再做 try/except。
"""

import os
import requests
from typing import Optional


# 后端 API 地址，优先从环境变量读取。
#
# ⚠️ 默认值刻意写成 127.0.0.1 而不是 localhost。
# 原因：Windows 上 localhost 会优先解析到 IPv6 的 ::1，而 uvicorn 绑的是
# 0.0.0.0（仅 IPv4）。于是每个请求都要先经历一次 IPv6 连接失败再回退到 IPv4，
# 实测单次 410ms；换成 127.0.0.1 后降到 8ms，差 50 倍。
# 而每次页面交互（Streamlit rerun）要打好几个接口，这个差距会被放大成
# "点什么都要等一秒"。Docker 部署时由 compose 注入 BACKEND_URL=http://backend:8000，
# 不受这个默认值影响。
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# 请求超时设置（秒）
TIMEOUT_SHORT = 10     # 普通查询
TIMEOUT_UPLOAD = 120   # 文件上传（含解析+向量化）
TIMEOUT_CHAT = 120     # 问答（含 LLM 生成）
TIMEOUT_COMPARE = 180  # 跨文献对比（检索轮次 = 论文数，模型输出也更长）


def _api_url(path: str) -> str:
    """拼接完整的 API URL"""
    return f"{BACKEND_URL}{path}"


def _error_response(message: str) -> dict:
    """构造统一的错误响应"""
    return {"code": 500, "message": message}


def _http_error(resp, fallback: str) -> dict:
    """
    把后端的 HTTP 错误体统一成前端使用的 {code, message}。

    FastAPI 的 HTTPException 返回 {"detail": "..."}，而业务接口返回
    {"code": ..., "message": ...}。页面只认 code/message，所以在这里抹平。
    """
    detail = fallback
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = body.get("detail") or body.get("message") or fallback
    except Exception:
        pass
    return {"code": getattr(resp, "status_code", 500), "message": str(detail)}


# ============================================================
# 系统相关
# ============================================================

def health_check() -> dict:
    """
    健康检查。

    后端约定：
        status: healthy（依赖都正常）/ degraded（服务活着但数据库不可用）
        database: connected / disconnected
        vector_store: loaded（内存已加载）/ on-disk（磁盘有索引但未加载）/ empty

    Returns:
        健康检查字典；连不上后端时返回 status=unhealthy
    """
    try:
        resp = requests.get(_api_url("/api/v1/health"), timeout=5)
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"status": "unhealthy", "database": "unknown", "vector_store": "unknown"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def is_backend_connected() -> bool:
    """后端进程是否活着（degraded 也算活着，只是数据库不可用）"""
    result = health_check()
    return result.get("status") in ("healthy", "degraded")


# ============================================================
# 文档管理
# ============================================================

def upload_document(file) -> dict:
    """
    上传文档（触发解析→切分→向量化全流程）。

    Args:
        file: Streamlit 的 UploadedFile 对象
    """
    try:
        files = {
            "file": (file.name, file.getvalue(), file.type or "application/octet-stream")
        }
        resp = requests.post(
            _api_url("/api/v1/documents/upload"),
            files=files,
            timeout=TIMEOUT_UPLOAD,
        )
        if not resp.ok:
            return _http_error(resp, "上传失败")
        return resp.json()
    except requests.exceptions.Timeout:
        return _error_response("上传超时：文件较大或模型推理较慢，请稍后重试")
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务，请确认已启动")
    except Exception as e:
        return _error_response(f"上传失败: {str(e)}")


def get_documents(page: int = 1, page_size: int = 50) -> dict:
    """获取文档列表"""
    try:
        resp = requests.get(
            _api_url("/api/v1/documents"),
            params={"page": page, "page_size": page_size},
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务")
    except Exception as e:
        return _error_response(f"获取文档列表失败: {str(e)}")


def get_document_detail(document_id: int) -> dict:
    """获取单个文档详情"""
    try:
        resp = requests.get(
            _api_url(f"/api/v1/documents/{document_id}"),
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except Exception as e:
        return _error_response(f"获取文档详情失败: {str(e)}")


def delete_document(document_id: int) -> dict:
    """删除文档"""
    try:
        resp = requests.delete(
            _api_url(f"/api/v1/documents/{document_id}"),
            timeout=TIMEOUT_SHORT,
        )
        if not resp.ok:
            return _http_error(resp, "删除失败")
        return resp.json()
    except Exception as e:
        return _error_response(f"删除失败: {str(e)}")


def get_document_stats() -> dict:
    """获取文档统计信息"""
    try:
        resp = requests.get(
            _api_url("/api/v1/documents/stats/overview"),
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except Exception as e:
        return _error_response(f"获取统计失败: {str(e)}")


def rebuild_index() -> dict:
    """
    用数据库里的文本块重建 FAISS 索引。

    切换嵌入模型后不必重新上传文件；耗时与文本块数量成正比，
    所以超时时间用 TIMEOUT_UPLOAD（120s）而不是普通查询的 10s。
    """
    try:
        resp = requests.post(
            _api_url("/api/v1/documents/rebuild-index"),
            timeout=TIMEOUT_UPLOAD,
        )
        if not resp.ok:
            return _http_error(resp, "重建索引失败")
        return resp.json()
    except requests.exceptions.Timeout:
        return _error_response("重建超时：文本块较多，可稍后重试或查看后端日志")
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务")
    except Exception as e:
        return _error_response(f"重建索引失败: {str(e)}")


# ============================================================
# 聊天问答
# ============================================================

def chat_with_kb(
    question: str,
    session_id: Optional[int] = None,
    top_k: int = 4,
) -> dict:
    """
    向论文库提问（RAG 问答）。

    Args:
        question: 用户问题
        session_id: 会话 ID（为空则自动创建新会话）
        top_k: 检索文档块数量
    """
    try:
        payload = {"question": question, "top_k": top_k}
        if session_id is not None:
            payload["session_id"] = session_id

        resp = requests.post(
            _api_url("/api/v1/chat"),
            json=payload,
            timeout=TIMEOUT_CHAT,
        )
        if not resp.ok:
            return _http_error(resp, "问答失败")
        return resp.json()
    except requests.exceptions.Timeout:
        return _error_response("回答生成超时，请稍后重试")
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务")
    except Exception as e:
        return _error_response(f"问答失败: {str(e)}")


def compare_papers(
    document_ids: list,
    question: Optional[str] = None,
    aspect: Optional[str] = None,
    chunks_per_doc: Optional[int] = None,
) -> dict:
    """
    跨文献对比：选多篇论文，按维度对齐后生成对比结论。

    与 chat_with_kb 的差别在语义上：那边是"问一个问题"，
    这边是"给我一张表"。后端会用每篇独立配额检索，避免长论文挤空短论文。

    Args:
        document_ids: 参与对比的论文 ID 列表（至少 2 个）
        question: 对比问题（可选，与 aspect 至少给一个）
        aspect: 对比维度（可选）
        chunks_per_doc: 每篇取几段（可选）
    """
    try:
        payload: dict = {"document_ids": list(document_ids)}
        if question:
            payload["question"] = question
        if aspect:
            payload["aspect"] = aspect
        if chunks_per_doc:
            payload["chunks_per_doc"] = chunks_per_doc

        resp = requests.post(
            _api_url("/api/v1/chat/compare"),
            json=payload,
            timeout=TIMEOUT_COMPARE,
        )
        if not resp.ok:
            return _http_error(resp, "对比失败")
        return resp.json()
    except requests.exceptions.Timeout:
        return _error_response("对比生成超时：论文较多或模型较慢，可减少对比篇数后重试")
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务")
    except Exception as e:
        return _error_response(f"对比失败: {str(e)}")


def get_sessions(page: int = 1, page_size: int = 50) -> dict:
    """获取聊天会话列表"""
    try:
        resp = requests.get(
            _api_url("/api/v1/chat/sessions"),
            params={"page": page, "page_size": page_size},
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except requests.exceptions.ConnectionError:
        return _error_response("无法连接后端服务")
    except Exception as e:
        return _error_response(f"获取会话列表失败: {str(e)}")


def get_session_messages(session_id: int) -> dict:
    """获取指定会话的所有消息"""
    try:
        resp = requests.get(
            _api_url(f"/api/v1/chat/sessions/{session_id}/messages"),
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except Exception as e:
        return _error_response(f"获取消息失败: {str(e)}")


def delete_session(session_id: int) -> dict:
    """删除指定会话"""
    try:
        resp = requests.delete(
            _api_url(f"/api/v1/chat/sessions/{session_id}"),
            timeout=TIMEOUT_SHORT,
        )
        return resp.json()
    except Exception as e:
        return _error_response(f"删除会话失败: {str(e)}")


# ============================================================
# 辅助函数
# ============================================================

def format_file_size(size_bytes: int) -> str:
    """将字节数格式化为人类可读的文件大小"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
