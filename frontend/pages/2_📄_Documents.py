"""
📄 文档管理页面

功能：
- 上传文档（PDF/TXT/MD），上传后自动触发 解析→切分→向量化
- 查看已上传文档列表，展示文件信息和处理状态
- 删除文档
- 论文库统计概览

所有操作通过 API 调用后端完成。
"""

import streamlit as st
from utils import (
    upload_document,
    get_documents,
    delete_document,
    get_document_stats,
    rebuild_index,
    health_check,
    format_file_size,
)

# ========== 页面配置 ==========
st.set_page_config(
    page_title="文档管理 - 论文阅读助手",
    page_icon="📄",
    layout="wide",
)

# ========== 自定义样式 ==========
st.markdown("""
<style>
    /* 文档卡片 */
    .doc-card {
        background: white;
        border: 1px solid #e8eaed;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: box-shadow 0.2s;
    }
    .doc-card:hover {
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .doc-name {
        font-weight: 600;
        color: #333;
        font-size: 0.95rem;
    }
    .doc-meta {
        color: #888;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }
    /* 状态标签 */
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-completed {
        background: #dcfce7;
        color: #166534;
    }
    .status-processing {
        background: #fef3c7;
        color: #92400e;
    }
    .status-failed {
        background: #fee2e2;
        color: #991b1b;
    }
    /* 统计卡片 */
    .stat-box {
        background: linear-gradient(145deg, #f8f9ff, #f0f2ff);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    .stat-box .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    .stat-box .stat-label {
        font-size: 0.8rem;
        color: #888;
        margin-top: 0.2rem;
    }
    /* 上传区域 */
    .upload-section {
        background: linear-gradient(145deg, #fafbff 0%, #f5f7ff 100%);
        border: 2px dashed #c7d0ea;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== 检查后端状态 ==========
status = health_check()
backend_ok = status.get("status") in ("healthy", "degraded")
db_ok = status.get("database") == "connected"

# ========== 侧边栏 ==========
with st.sidebar:
    st.markdown("### 📄 文档管理")
    st.markdown("---")

    if backend_ok:
        st.markdown(
            '<span style="color: #22c55e; font-weight: 600;">● 后端已连接</span>',
            unsafe_allow_html=True,
        )

        # 统计信息
        stats_resp = get_document_stats()
        if stats_resp.get("code") == 200 and stats_resp.get("data"):
            stats = stats_resp["data"]
            st.markdown("---")
            st.markdown("**📊 论文库概览**")
            st.metric("文档总数", stats.get("total_documents", 0))
            st.metric("文本块数", stats.get("total_chunks", 0))
            st.metric("向量总数", stats.get("total_vectors", 0))

            # 状态分布
            completed = stats.get("completed", 0)
            processing = stats.get("processing", 0)
            failed = stats.get("failed", 0)
            if completed or processing or failed:
                st.markdown("---")
                st.caption(f"✅ 已完成: {completed}　⏳ 处理中: {processing}　❌ 失败: {failed}")
    else:
        st.markdown(
            '<span style="color: #ef4444; font-weight: 600;">● 后端未连接</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**支持的文件格式**")
    st.markdown("- 📕 PDF 文档")
    st.markdown("- 📝 TXT 文本")
    st.markdown("- 📘 Markdown")

    # 维护操作：重建索引（换嵌入模型 / 索引漂移时使用）
    if backend_ok:
        st.markdown("---")
        with st.expander("🔧 维护"):
            st.caption("用数据库中的文本块重建向量索引；会重新做一次 embedding，耗时与块数成正比。")
            if st.button("重建向量索引", use_container_width=True):
                with st.spinner("正在重建索引，请勿关闭页面..."):
                    rb = rebuild_index()
                if rb.get("code") == 200:
                    st.success(rb.get("message", "重建完成"))
                    st.rerun()
                else:
                    st.error(rb.get("message", "重建失败"))


# ========== 主区域 ==========
st.markdown("## 📄 文档管理")

if not backend_ok:
    st.error("⚠️ 后端服务未连接，请确认后端已启动后刷新页面。")
    st.info("启动命令：`python -m backend.main`")
    st.stop()

# 后端活着但数据库不可用：上传/列表/删除都会失败，先把原因说清楚
if not db_ok:
    st.warning(
        "⚠️ 后端无法连接数据库（health status = degraded）。"
        "请检查 MySQL 是否运行、`.env` 中的连接配置是否正确。"
    )

# ----------------------------------------------------------
# 📤 文档上传区
# ----------------------------------------------------------
st.markdown("### 📤 上传文档")

uploaded_file = st.file_uploader(
    "选择文件上传到论文库",
    type=["pdf", "txt", "md"],
    help="支持 PDF、TXT、Markdown 格式。上传后将自动解析、切分、向量化。",
    label_visibility="collapsed",
)

if uploaded_file is not None:
    # 显示文件预览信息
    col1, col2, col3 = st.columns(3)
    with col1:
        file_icon = {"pdf": "📕", "txt": "📝", "md": "📘"}.get(
            uploaded_file.name.rsplit(".", 1)[-1].lower(), "📄"
        )
        st.markdown(f"**{file_icon} 文件名**：{uploaded_file.name}")
    with col2:
        st.markdown(f"**📏 大小**：{format_file_size(uploaded_file.size)}")
    with col3:
        file_type = uploaded_file.name.rsplit(".", 1)[-1].upper()
        st.markdown(f"**📋 类型**：{file_type}")

    # 上传按钮
    if st.button("🚀 开始上传并处理", type="primary", use_container_width=True):
        # 进度展示
        progress_bar = st.progress(0, text="正在上传文件...")

        progress_bar.progress(20, text="📤 文件上传中...")
        result = upload_document(uploaded_file)
        progress_bar.progress(60, text="🔄 正在解析和向量化...")

        if result.get("code") == 200:
            progress_bar.progress(100, text="✅ 处理完成!")
            st.success(f"✅ {result.get('message', '上传成功')}")

            # 展示处理结果
            if result.get("data"):
                data = result["data"]
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("文档 ID", data.get("id", "-"))
                with r2:
                    st.metric("文本块数", data.get("chunk_count", 0))
                with r3:
                    st.metric("状态", data.get("status", "-"))

            st.balloons()
        else:
            progress_bar.progress(100, text="❌ 处理失败")
            error_msg = result.get("message", "未知错误")
            if "detail" in result:
                error_msg = result["detail"]
            st.error(f"❌ {error_msg}")

st.markdown("---")

# ----------------------------------------------------------
# 📋 文档列表区
# ----------------------------------------------------------
st.markdown("### 📋 已上传文档")

# 刷新按钮
col_refresh, col_spacer = st.columns([1, 5])
with col_refresh:
    if st.button("🔄 刷新", use_container_width=True):
        st.rerun()

# 获取文档列表
result = get_documents(page=1, page_size=50)

if result.get("code") == 200 and result.get("data"):
    documents = result["data"].get("documents", [])
    total = result["data"].get("total", 0)

    st.caption(f"共 {total} 个文档")

    if documents:
        for doc in documents:
            doc_id = doc.get("id")
            filename = doc.get("filename", "未知")
            file_type = doc.get("file_type", "")
            file_size = doc.get("file_size", 0)
            chunk_count = doc.get("chunk_count", 0)
            doc_status = doc.get("status", "unknown")
            error_msg = doc.get("error_message")
            created_at = doc.get("created_at", "")

            # 文件图标
            icon = {"pdf": "📕", "txt": "📝", "md": "📘"}.get(file_type, "📄")

            # 状态标签
            status_class = {
                "completed": "status-completed",
                "processing": "status-processing",
                "failed": "status-failed",
            }.get(doc_status, "")
            status_text = {
                "completed": "✅ 已完成",
                "processing": "⏳ 处理中",
                "failed": "❌ 失败",
            }.get(doc_status, doc_status)

            # 文档行
            cols = st.columns([4, 1.5, 1.5, 1.5, 1])

            with cols[0]:
                st.markdown(f"**{icon} {filename}**")
                # 上传时间之前取了却没用上，这里补上（后端返回的是 UTC 时间串）
                if created_at:
                    st.caption(f"🕒 {str(created_at).replace('T', ' ')[:19]}")
                if error_msg:
                    st.caption(f"⚠️ {error_msg[:50]}")

            with cols[1]:
                st.caption(f"📏 {format_file_size(file_size)}")

            with cols[2]:
                st.caption(f"🧩 {chunk_count} 块")

            with cols[3]:
                st.markdown(
                    f'<span class="status-badge {status_class}">{status_text}</span>',
                    unsafe_allow_html=True,
                )

            with cols[4]:
                if st.button("🗑️", key=f"del_{doc_id}", help=f"删除 {filename}"):
                    del_result = delete_document(doc_id)
                    if del_result.get("code") == 200:
                        st.success(f"已删除: {filename}")
                        st.rerun()
                    else:
                        st.error(del_result.get("message", "删除失败"))

            st.divider()
    else:
        st.info("📭 论文库为空。请上传你的第一个文档！")
else:
    error_msg = result.get("message", "无法获取文档列表")
    st.error(f"❌ {error_msg}")
