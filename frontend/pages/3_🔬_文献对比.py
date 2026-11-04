"""
🔬 跨文献对比页面

针对"我需要同时看几篇论文，比较它们的方法差异"这个真实场景。

和 💬 论文问答 的区别（也是这个页面的核心价值）：
    问答是"在整个库里找最相关的几段"，结果可能全部落在同一篇论文里；
    对比必须让每篇论文都有自己的配额，否则短论文在表格里全是"未提及"。

页面流程：
1. 选择 2~N 篇已完成的论文
2. 选择/输入对比维度（方法 / 实验 / 创新点 …）
3. 后端按维度推导目标章节，逐篇独立检索 → 篇内还原阅读顺序 → 结构化对比
4. 展示对比表 + 关键差异 + 每篇的引用片段（带 [A-2] 编号，可回溯到页码）
"""

import streamlit as st

from utils import compare_papers, get_documents, health_check

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

# 预置的对比维度：文案里带"方法/实验/创新"等关键词，
# 后端 preferred_sections() 会据此推导要对齐的章节。
ASPECT_PRESETS = [
    "方法（各自怎么实现的）",
    "实验设置与评测结果",
    "创新点与贡献",
    "局限性",
    "自定义",
]

st.set_page_config(
    page_title="跨文献对比 - 论文阅读助手",
    page_icon="🔬",
    layout="wide",
)

# ========== 自定义样式 ==========
st.markdown("""
<style>
    .doc-badge {
        display: inline-block;
        background: #eef1ff;
        color: #4c51bf;
        border-radius: 6px;
        padding: 0.1rem 0.5rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .chunk-card {
        background: #fbfbff;
        border-left: 3px solid #a5b4fc;
        border-radius: 0 8px 8px 0;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .chunk-card .chunk-head {
        color: #6366f1;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .chunk-card .chunk-meta {
        color: #9aa0a6;
        font-size: 0.75rem;
        margin-top: 0.35rem;
    }
</style>
""", unsafe_allow_html=True)


# ========== 后端状态 ==========
status = health_check()
backend_ok = status.get("status") in ("healthy", "degraded")

with st.sidebar:
    st.markdown("### 🔬 跨文献对比")
    st.markdown("---")
    st.caption(
        "选 2 篇以上论文，按同一维度对齐后生成对比表。"
        "每篇论文独立检索配额，篇内按原文顺序还原。"
    )
    st.markdown("---")
    st.markdown("**💡 建议**")
    st.caption("对比同主题的论文效果最好；跨主题对比时，表格里大量「未提及」是正常的。")


st.markdown("## 🔬 跨文献对比")

if not backend_ok:
    st.error("⚠️ 后端服务未连接，请确认后端已启动后刷新页面。")
    st.info("启动命令：`python -m backend.main`")
    st.stop()


# ========== 1. 选择论文 ==========
st.markdown("### 1️⃣ 选择论文")

doc_resp = get_documents(page=1, page_size=100)
documents = []
if doc_resp.get("code") == 200 and doc_resp.get("data"):
    documents = [
        d for d in doc_resp["data"].get("documents", [])
        if d.get("status") == "completed"
    ]

if len(documents) < 2:
    st.warning(
        f"当前论文库只有 {len(documents)} 篇处理完成的论文，至少需要 2 篇才能对比。"
        "请先到 📄 文档管理 上传论文。"
    )
    st.stop()

# 文件名做展示、id 做值：Streamlit 的 multiselect 需要可哈希的选项，
# 用 (id, filename) 元组最直观，也避免同名文件互相覆盖。
option_map = {f"{d['filename']}  ·  {d.get('chunk_count', 0)} 块": d["id"] for d in documents}
picked_labels = st.multiselect(
    "选择要对比的论文",
    options=list(option_map.keys()),
    default=list(option_map.keys())[:2],
    help="建议 2~4 篇；超过 4 篇上下文会被摊薄，结论质量反而下降",
)
picked_ids = [option_map[label] for label in picked_labels]

if picked_ids and len(picked_ids) < 2:
    st.info("请至少选择 2 篇论文。")


# ========== 2. 选择对比维度 ==========
st.markdown("### 2️⃣ 对比维度")

col_aspect, col_num = st.columns([3, 2])

with col_aspect:
    preset = st.selectbox(
        "按什么维度对比",
        options=ASPECT_PRESETS,
        index=0,
        help="维度会决定后端对齐哪个章节，例如选「实验设置」就只对齐实验章",
    )
    if preset == "自定义":
        aspect = st.text_input(
            "自定义维度",
            placeholder="例如：数据预处理流程 / 训练策略 / 部署成本",
        )
    else:
        aspect = preset

with col_num:
    chunks_per_doc = st.slider(
        "每篇取几段",
        min_value=2,
        max_value=8,
        value=4,
        help="每篇配额。调大会看到更完整的方法章，但也更容易超出上下文预算被截断",
    )

question = st.text_input(
    "补充问题（可选）",
    placeholder="例如：这几篇在损失函数设计上的差异是什么？",
    help="填了会用这句话去检索，比只用维度词更精准",
)


# ========== 3. 执行对比 ==========
st.markdown("### 3️⃣ 开始对比")

run = st.button(
    "🔍 生成对比",
    type="primary",
    disabled=len(picked_ids) < 2,
    use_container_width=True,
)

if run:
    with st.spinner("正在逐篇检索并对齐章节，然后生成对比…"):
        result = compare_papers(
            document_ids=picked_ids,
            question=question or None,
            aspect=aspect,
            chunks_per_doc=chunks_per_doc,
        )
    # 结果放 session_state，避免 Streamlit 重跑时丢失
    st.session_state["compare_result"] = result


# ========== 4. 渲染结果 ==========
result = st.session_state.get("compare_result")

if result:
    if result.get("code") != 200 or not result.get("data"):
        message = result.get("message") or result.get("detail") or "对比失败"
        st.error(f"❌ {message}")
        st.stop()

    data = result["data"]
    stats = data.get("stats") or {}
    groups = data.get("groups") or []

    st.markdown("---")

    # ---- 运行指标：这些数字决定了对比结论可不可信 ----
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("对比论文数", stats.get("document_count", len(groups)))
    with c2:
        st.metric("参与片段", stats.get("total_chunks", 0))
    with c3:
        st.metric("上下文长度", f"{stats.get('context_chars', 0):,}")
    with c4:
        elapsed = data.get("response_time")
        st.metric("总耗时", f"{elapsed}s" if elapsed is not None else "—")

    # 章节对齐情况：让用户知道这次用的是哪一章做的对比
    sections = stats.get("sections") or []
    if sections:
        labels = "、".join(SECTION_LABELS.get(s, s) for s in sections)
        st.caption(f"🎯 本次对齐章节：**{labels}**")

    # 截断提示：这是最容易导致「结论不完整」的原因，必须显式告知
    truncated = stats.get("truncated_documents") or []
    if truncated:
        st.warning(
            "⚠️ 以下论文因超出上下文预算被截断，对比结论可能不完整："
            + "、".join(truncated)
            + "。可以调小「每篇取几段」或减少对比篇数。"
        )

    # ---- 对比结论 ----
    st.markdown("### 📊 对比结论")
    st.markdown(data.get("answer", ""))

    # ---- 每篇的引用片段（可回溯） ----
    st.markdown("---")
    st.markdown("### 📎 各论文引用片段")
    st.caption("表格里的 [A-2] 就对应下面对应编号的片段，可据此回原文核对。")

    letters = [g.get("letter", "?") for g in groups]
    tabs = st.tabs([f"论文{letter}" for letter in letters]) if groups else []

    for tab, group in zip(tabs, groups):
        with tab:
            st.markdown(f"**{group.get('document_name', '未知')}**")
            chunks = group.get("chunks") or []
            if not chunks:
                st.info("这篇论文在目标章节里没有检索到可用片段。")
                continue

            for chunk in chunks:
                meta_parts = []
                if chunk.get("page") is not None:
                    meta_parts.append(f"第 {chunk['page']} 页")
                section_label = SECTION_LABELS.get(chunk.get("section") or "", "")
                if section_label:
                    meta_parts.append(section_label)
                if chunk.get("chunk_index") is not None:
                    meta_parts.append(f"块 #{chunk['chunk_index']}")
                rerank_score = chunk.get("rerank_score")
                if rerank_score is not None:
                    meta_parts.append(f"重排 {rerank_score:.0%}")

                content = chunk.get("chunk_content", "")
                st.markdown(
                    f"""
                    <div class="chunk-card">
                        <div class="chunk-head">[{chunk.get('ref', '?')}]</div>
                        <div>{content[:400]}{'…' if len(content) > 400 else ''}</div>
                        <div class="chunk-meta">{' · '.join(meta_parts)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
