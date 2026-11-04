"""
Streamlit 前端 — 首页

展示项目介绍、系统状态、论文库统计，并作为多页面应用的导航入口。
"""

import streamlit as st
from utils import health_check, get_document_stats


def main():
    # ========== 页面配置 ==========
    st.set_page_config(
        page_title="论文阅读助手",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ========== 自定义样式 ==========
    st.markdown("""
    <style>
        /* 渐变标题 */
        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0;
            line-height: 1.2;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #6c757d;
            margin-top: 0.3rem;
            margin-bottom: 2rem;
        }
        /* 功能卡片 */
        .feature-card {
            background: linear-gradient(145deg, #f8f9ff 0%, #f0f2ff 100%);
            border-radius: 16px;
            padding: 1.5rem;
            border-left: 4px solid #667eea;
            margin-bottom: 0.8rem;
            transition: transform 0.2s;
        }
        .feature-card:hover {
            transform: translateY(-2px);
        }
        .feature-card h3 {
            margin: 0 0 0.5rem 0;
            color: #333;
            font-size: 1.1rem;
        }
        .feature-card p {
            margin: 0;
            color: #555;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        /* 状态指示器 */
        .status-dot {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 6px;
        }
        .status-green { background-color: #22c55e; }
        .status-red { background-color: #ef4444; }
        .status-yellow { background-color: #f59e0b; }
        /* 步骤编号 */
        .step-num {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            font-weight: 700;
            font-size: 0.85rem;
            margin-right: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ========== 侧边栏 ==========
    with st.sidebar:
        st.markdown("### 📚 论文阅读助手")
        st.caption("v1.1.0")
        st.markdown("---")

        # 系统状态
        # healthy = 依赖都正常；degraded = 后端活着但数据库不可用
        status = health_check()
        backend_up = status.get("status") in ("healthy", "degraded")
        is_healthy = status.get("status") == "healthy"
        db_ok = status.get("database") == "connected"
        vs_status = status.get("vector_store", "unknown")

        st.markdown("**🔌 系统状态**")

        if backend_up and is_healthy:
            st.markdown(
                '<span class="status-dot status-green"></span> 服务运行中',
                unsafe_allow_html=True,
            )
        elif backend_up:
            st.markdown(
                '<span class="status-dot status-yellow"></span> 服务运行中（依赖异常）',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-dot status-red"></span> 服务未连接',
                unsafe_allow_html=True,
            )

        if db_ok:
            st.markdown(
                '<span class="status-dot status-green"></span> 数据库已连接',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="status-dot status-red"></span> 数据库未连接',
                unsafe_allow_html=True,
            )

        # vector_store: loaded（内存已加载）/ on-disk（磁盘有索引但本次未加载）/ empty
        if vs_status == "loaded":
            vs_color, vs_label = "green", "向量库已加载"
        elif vs_status == "on-disk":
            vs_color, vs_label = "yellow", "向量库未加载（磁盘已有索引）"
        else:
            vs_color, vs_label = "yellow", "向量库为空"
        st.markdown(
            f'<span class="status-dot status-{vs_color}"></span> {vs_label}',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**📖 功能导航**")
        st.page_link("pages/1_💬_Chat.py", label="💬 论文问答", icon="💬")
        st.page_link("pages/3_🔬_文献对比.py", label="🔬 跨文献对比", icon="🔬")
        st.page_link("pages/2_📄_Documents.py", label="📄 文档管理", icon="📄")

    # ========== 主区域 — 标题 ==========
    st.markdown(
        '<p class="main-title">📚 论文阅读助手</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">章节感知检索 + 跨文献对比 · 回答附带页码与章节引用</p>',
        unsafe_allow_html=True,
    )

    # ========== 论文库统计（仅在后端正常时显示） ==========
    if is_healthy:
        stats_resp = get_document_stats()
        if stats_resp.get("code") == 200:
            stats = stats_resp.get("data", {})
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("📄 论文数", stats.get("total_documents", 0))
            with c2:
                st.metric("🧩 文本块数", stats.get("total_chunks", 0))
            with c3:
                st.metric("📐 向量总数", stats.get("total_vectors", 0))
            with c4:
                st.metric("✅ 已完成", stats.get("completed", 0))

    st.markdown("---")

    # ========== 核心能力 ==========
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🏷️ 章节感知检索</h3>
            <p>入库时自动识别摘要/引言/方法/实验等章节，
            按问题意图对齐目标章节，问「创新点」不再检到方法章。</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🔬 跨文献对比</h3>
            <p>多篇论文按同一维度生成对比表。
            每篇独立检索配额，篇内还原原文顺序，避免长论文挤空短论文。</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📎 引用可溯源</h3>
            <p>每条结论标注来源论文、页码与章节；
            对比表里的 [A-2] 可直接回查到具体片段。</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ========== 快速开始 ==========
    st.markdown("### 🚀 快速开始")

    st.markdown("""
    <p>
        <span class="step-num">1</span> 进入 <strong>📄 文档管理</strong>，上传你的论文 PDF<br/>
        <span class="step-num">2</span> 系统自动完成 解析 → 章节标注 → 切分 → 向量化<br/>
        <span class="step-num">3</span> 到 <strong>💬 论文问答</strong> 问单篇问题，或到 <strong>🔬 跨文献对比</strong> 做横向比较
    </p>
    """, unsafe_allow_html=True)

    # ========== 技术栈 ==========
    st.markdown("---")
    st.markdown("### 🛠️ 技术栈")
    t1, t2, t3, t4, t5 = st.columns(5)
    with t1:
        st.markdown("**前端**\n\nStreamlit")
    with t2:
        st.markdown("**后端**\n\nFastAPI")
    with t3:
        st.markdown("**AI 框架**\n\nLangChain")
    with t4:
        st.markdown("**检索**\n\nFAISS + bge-reranker")
    with t5:
        st.markdown("**数据库**\n\nMySQL 8.0")


if __name__ == "__main__":
    main()
