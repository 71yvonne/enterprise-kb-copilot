"""
💬 论文问答页面

功能：
- 左侧边栏：会话列表，新建/切换/删除会话
- 主区域：聊天对话界面，展示问答和引用来源
- 底部：问题输入框

使用 Streamlit 原生 chat 组件构建对话 UI。
所有逻辑通过 API 调用后端完成。
"""

import streamlit as st
from utils import (
    chat_with_kb,
    get_sessions,
    get_session_messages,
    delete_session,
    health_check,
)

# 章节 key → 中文展示名（与后端 section_detector.SECTION_LABELS 保持一致）
SECTION_LABELS = {
    "abstract": "摘要",
    "introduction": "引言",
    "related_work": "相关工作",
    "method": "方法",
    "experiment": "实验",
    "conclusion": "结论",
    "other": "正文",
}

# ========== 页面配置 ==========
st.set_page_config(
    page_title="论文问答 - 论文阅读助手",
    page_icon="💬",
    layout="wide",
)

# ========== 自定义样式 ==========
st.markdown("""
<style>
    /* 引用来源卡片 */
    .source-card {
        background: #f8f9ff;
        border-left: 3px solid #667eea;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .source-card .source-title {
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.3rem;
    }
    .source-card .source-content {
        color: #555;
        line-height: 1.5;
    }
    .source-card .source-meta {
        color: #999;
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }
    /* 会话按钮 */
    .session-item {
        padding: 0.4rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    /* 空状态提示 */
    .empty-hint {
        text-align: center;
        color: #999;
        padding: 3rem 1rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ========== Session State 初始化 ==========
# st.session_state 绑定在浏览器连接上：按 F5 会新建连接、状态全部丢失。
# 因此把当前会话 ID 同步到 URL 查询参数（?s=5），刷新时再读回来。
URL_SESSION_KEY = "s"


def _read_session_from_url():
    """从 URL 查询参数读取会话 ID，非法值返回 None"""
    try:
        raw = st.query_params.get(URL_SESSION_KEY)
    except Exception:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _write_session_to_url(session_id):
    """把会话 ID 写进 URL；None 表示新对话，清掉参数"""
    try:
        if session_id is None:
            if URL_SESSION_KEY in st.query_params:
                del st.query_params[URL_SESSION_KEY]
        else:
            st.query_params[URL_SESSION_KEY] = str(session_id)
    except Exception:
        # 查询参数不可用时静默降级，不影响聊天主流程
        pass


if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = _read_session_from_url()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sessions_cache" not in st.session_state:
    st.session_state.sessions_cache = []


def load_sessions():
    """从后端加载会话列表"""
    resp = get_sessions(page=1, page_size=50)
    if resp.get("code") == 200 and resp.get("data"):
        st.session_state.sessions_cache = resp["data"].get("sessions", [])
    else:
        st.session_state.sessions_cache = []


def load_session_messages(session_id: int):
    """从后端加载指定会话的历史消息"""
    resp = get_session_messages(session_id)
    if resp.get("code") == 200 and resp.get("data"):
        raw_messages = resp["data"].get("messages", [])
        # 转换为 Streamlit 的消息格式
        messages = []
        for msg in raw_messages:
            messages.append({"role": "user", "content": msg["question"]})
            messages.append({
                "role": "assistant",
                "content": msg["answer"],
                "sources": msg.get("sources", []),
                "response_time": msg.get("response_time"),
            })
        st.session_state.messages = messages
    else:
        st.session_state.messages = []


def start_new_chat():
    """开始新对话"""
    st.session_state.current_session_id = None
    st.session_state.messages = []
    _write_session_to_url(None)


def switch_session(session_id: int):
    """切换到指定会话"""
    st.session_state.current_session_id = session_id
    load_session_messages(session_id)
    _write_session_to_url(session_id)


# 刷新页面后从 URL 恢复会话：把历史消息拉回来，接着聊
if st.session_state.current_session_id is not None and not st.session_state.messages:
    load_session_messages(st.session_state.current_session_id)
    if not st.session_state.messages:
        # URL 里记的会话已经不存在（被删除或清库），退回新对话
        st.session_state.current_session_id = None
        _write_session_to_url(None)


def render_sources(sources: list):
    """渲染引用来源卡片"""
    if not sources:
        return

    with st.expander(f"📎 引用来源（{len(sources)} 条）", expanded=False):
        for i, source in enumerate(sources, 1):
            doc_name = source.get("document_name", "未知文档")
            content = source.get("chunk_content", "")
            page = source.get("page")
            score = source.get("relevance_score")

            # 构建元信息
            meta_parts = []
            if page is not None:
                # 后端入库时已经把 PyPDFLoader 的 0 起页码转成 1 起，这里不再 +1
                meta_parts.append(f"📄 第 {page} 页")
            sec_label = SECTION_LABELS.get(source.get("section") or "", "")
            if sec_label:
                meta_parts.append(f"🏷️ {sec_label}")
            # 开了重排就以重排分为主（排序依据），向量分只在没有重排时展示
            rerank_score = source.get("rerank_score")
            if rerank_score is not None:
                meta_parts.append(f"重排 {rerank_score:.0%}")
            elif score is not None:
                meta_parts.append(f"向量相关度 {score:.0%}")
            meta_str = " · ".join(meta_parts)

            st.markdown(f"""
            <div class="source-card">
                <div class="source-title">📌 来源 {i}: {doc_name}</div>
                <div class="source-content">{content[:250]}{'...' if len(content) > 250 else ''}</div>
                <div class="source-meta">{meta_str}</div>
            </div>
            """, unsafe_allow_html=True)


# ========== 检查后端状态 ==========
# degraded = 后端活着但数据库不可用；仍然允许打开页面，只是提示库有问题
status = health_check()
backend_ok = status.get("status") in ("healthy", "degraded")
db_ok = status.get("database") == "connected"

# ========== 侧边栏：会话管理 ==========
with st.sidebar:
    st.markdown("### 💬 论文问答")
    st.markdown("---")

    # 新建对话按钮
    if st.button("➕ 新建对话", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()

    st.markdown("---")

    # 问答设置
    st.markdown("**⚙️ 设置**")
    # 默认 8：一篇论文通常有几十个块，只取 4 个时"总结方法"这类问题会严重缺料
    top_k = st.slider("检索数量 (top_k)", min_value=1, max_value=10, value=8, help="检索最相关的文档块数量；问全局性问题（总结方法/结论）时调大")

    st.markdown("---")

    # 会话列表
    st.markdown("**📋 历史会话**")

    if backend_ok:
        load_sessions()
        sessions = st.session_state.sessions_cache

        if sessions:
            for sess in sessions:
                sess_id = sess.get("id")
                title = sess.get("title", "未命名对话")
                count = sess.get("message_count", 0)

                # 当前选中的会话高亮
                is_current = sess_id == st.session_state.current_session_id

                col_btn, col_del = st.columns([5, 1])
                with col_btn:
                    label = f"{'▶ ' if is_current else ''}{title[:20]}"
                    if st.button(
                        label,
                        key=f"sess_{sess_id}",
                        use_container_width=True,
                        disabled=is_current,
                    ):
                        switch_session(sess_id)
                        st.rerun()

                with col_del:
                    if st.button("🗑", key=f"del_sess_{sess_id}", help="删除此会话"):
                        delete_session(sess_id)
                        if st.session_state.current_session_id == sess_id:
                            start_new_chat()
                        st.rerun()
        else:
            st.caption("暂无历史会话")
    else:
        st.warning("后端未连接")


# ========== 主区域：聊天界面 ==========
st.markdown("## 💬 论文问答")

if not backend_ok:
    st.error("⚠️ 后端服务未连接，请确认后端已启动后刷新页面。")
    st.info("启动命令：`python -m backend.main`")
    st.stop()

# 后端活着但数据库挂了：能打开页面，但问答/会话一定会失败，先把原因摆出来
if not db_ok:
    st.warning(
        "⚠️ 后端无法连接数据库（health status = degraded）。"
        "请检查 MySQL 是否运行、`.env` 中的 MYSQL_HOST / MYSQL_PORT / MYSQL_PASSWORD 是否正确。"
    )

# 当前会话标识
if st.session_state.current_session_id:
    st.caption(f"📌 当前会话 ID: {st.session_state.current_session_id}")
else:
    st.caption("📌 新对话（发送第一条消息后自动创建会话）")

st.markdown("---")

# 展示历史消息
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-hint">
        <p style="font-size: 2rem; margin-bottom: 0.5rem;">🤖</p>
        <p>你好！我是论文阅读助手。</p>
        <p>在下方提问，我会基于你上传的论文作答，并标注页码与章节。</p>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # AI 回复时展示引用来源
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg["sources"])

        # 展示响应时间
        if msg["role"] == "assistant" and msg.get("response_time"):
            st.caption(f"⏱️ 响应耗时: {msg['response_time']}s")


# ========== 用户输入框 ==========
if prompt := st.chat_input("请输入你的问题..."):
    # 立即显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用后端 RAG 接口
    with st.chat_message("assistant"):
        with st.spinner("🔍 正在检索论文库并生成回答..."):
            result = chat_with_kb(
                question=prompt,
                session_id=st.session_state.current_session_id,
                top_k=top_k,
            )

        if result.get("code") == 200 and result.get("data"):
            data = result["data"]
            answer = data["answer"]
            sources = data.get("sources", [])
            session_id = data.get("session_id")
            # 后端现在会返回本次耗时（ChatResponseData.response_time），
            # 新消息也能显示 ⏱️ 了，不再只在历史消息里有。
            response_time = data.get("response_time")

            # 更新当前会话 ID（首次消息时后端自动创建会话），并写进 URL
            if session_id and not st.session_state.current_session_id:
                st.session_state.current_session_id = session_id
                _write_session_to_url(session_id)

            # 展示回答
            st.markdown(answer)

            # 展示引用来源
            if sources:
                render_sources(sources)

            # 保存到消息历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "response_time": response_time,
            })

        else:
            error_msg = result.get("message", "未知错误")
            if "detail" in result:
                error_msg = result["detail"]
            st.error(f"❌ {error_msg}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"❌ 回答失败: {error_msg}",
            })
