"""
聊天服务 — 论文问答与跨文献对比

两条主链路：

一、单篇问答（ask）
    问题 → 章节意图识别 → 向量召回 → cross-encoder 精排 → 构建上下文
    → LLM 生成 → 保存记录 → 返回回答 + 引用（文件/页码/章节/分数）

二、跨文献对比（compare）
    对比 **不能** 复用 ask 的"全库 top-k"，否则一篇长论文会把其他论文挤空，
    结果变成"一篇的摘要 + 三篇的零碎"。因此 compare 改为：
    每篇论文分配独立配额 → 按目标章节对齐 → 篇内按 chunk_index 还原
    阅读顺序 → 按字符预算截断 → 带 [A-2] 式编号交给模型做结构化对比。

两条链路的共同点：只依据检索到的片段作答，片段不足时明确拒答。
"""

import threading
import time
from typing import Optional
from sqlalchemy.orm import Session

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from backend.config import get_settings
from backend.models import ChatSession, ChatMessage, Document
from backend.services.vector_service import VectorService
from backend.services.rerank_service import RerankService
from backend.utils.logger import logger
from backend.utils.section_detector import preferred_sections, section_label


# ============================================================
# 论文问答系统提示词
# ============================================================
RAG_SYSTEM_PROMPT = """你是「论文阅读助手」，帮研究者快速读懂学术论文。

你的任务：仅依据下方【参考片段】回答用户关于论文的问题。

规则：
1. **只依据片段**：片段来自用户上传的论文，可能只覆盖其中一部分。
   不要用你自己的背景知识补全，也不要替作者"完善"论证。
2. **标明出处**：每条结论后用 [来源i] 标注取自哪一段，i 是片段编号。
   没有片段支撑的话不要写。
3. **区分事实与主张**：作者"做了什么"（方法、超参、数据集）和
   作者"声称什么"（贡献、效果）要分开表述，不要把作者的主张写成既定事实。
4. **数字照抄**：指标、数据集名、超参数、模型规模一律照抄原文，
   不要换算、不要"约等于"、不要四舍五入。
5. **诚实告知**：片段不足以回答时，直接说"检索到的片段不足以回答这个问题"，
   并指出缺哪部分（例如"缺实验章节的内容"）。
   不要断言论文没有某个内容，也不要建议用户"上传完整版本"。
6. **格式**：中文回答，条理清晰；讲方法流程时按原文顺序列出步骤。

示例（格式参考，内容不重要）：
- 作者采用两阶段生成流程 [来源1]
- Stage-I 以 Canny 边缘检测结果作为控制条件 [来源2]
- 消融显示去掉车道标注后 FID 上升 [来源3]

【参考片段】
{context}"""


# ============================================================
# 跨文献对比系统提示词
# ============================================================
COMPARE_SYSTEM_PROMPT = """你是「论文阅读助手」的跨文献对比模式。

下方【各论文片段】按【论文A】【论文B】… 分段给出，每段带 [A-1] 这样的编号。
你的任务是把它们对齐，做结构化对比。

规则：
1. **只用给定片段**，不得引入外部知识或凭印象补充。
2. 先输出一张 Markdown 对比表：
   - 每一行对应一篇论文（列首写论文编号与文件名）
   - 列为用户关心的维度；用户没指定维度时用：
     核心问题 / 方法 / 关键设计 / 数据与实验 / 主要结论
3. 表格之后列 3–5 条「关键差异」：只写**真正不同**的地方。
   相同的点合并成一句话，不要逐篇复述表格。
4. 每格内容后标注出处，格式 [A-2]（论文A的第2个片段）。
5. 某篇论文的片段没有覆盖某个维度时，该格写「未提及」，
   **不要推测，也不要拿其他论文的内容去填补**。
6. 若所有论文的片段都不足以支撑某个维度，就在表格下方说明
   "该维度信息不足，建议换一个对比维度或补充相关章节"。
7. 表格单元格内不要换行；中文回答。

【各论文片段】
{context}"""


# ============================================================
# LLM 客户端（进程内单例）
# ============================================================
_llm_client: Optional[ChatOpenAI] = None
_llm_lock = threading.Lock()


def get_llm_client() -> ChatOpenAI:
    """
    按需创建并复用 LLM 客户端。

    原来每个请求都 new 一个 ChatOpenAI，会重复建连接池、重复打日志，
    高并发下纯属浪费。这里用双重检查锁做进程内单例。
    """
    global _llm_client
    if _llm_client is None:
        with _llm_lock:
            if _llm_client is None:
                settings = get_settings()
                _llm_client = ChatOpenAI(
                    model=settings.llm_model_name,
                    api_key=settings.openai_api_key,
                    base_url=settings.openai_base_url,
                    # 论文问答是事实型任务，温度调低以减少"自由发挥"；
                    # 需要发散时（如头脑风暴综述角度）再单独调高。
                    temperature=settings.llm_temperature,
                    max_tokens=2000,    # 限制单次回答长度，控制成本
                )
                logger.info(
                    "LLM 客户端已初始化: model='{}', base_url='{}', temperature={}",
                    settings.llm_model_name, settings.openai_base_url,
                    settings.llm_temperature,
                )
    return _llm_client


class ChatService:
    """论文问答与跨文献对比服务"""

    def __init__(self, db: Session):
        """
        初始化聊天服务。

        Args:
            db: 数据库会话（由 FastAPI 依赖注入提供）
        """
        self.db = db
        self.settings = get_settings()

        # 向量服务是进程级单例；若启动时已预热，这里不会再有模型加载开销
        self.vector_service = VectorService.get_instance()

    # ==========================================================
    # 一、单篇问答
    # ==========================================================
    def ask(
        self,
        question: str,
        session_id: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        RAG 问答：检索 → 构建上下文 → LLM 生成回答。

        Args:
            question: 用户问题
            session_id: 会话 ID（为空则自动创建新会话）
            top_k: 检索返回的文本块数量；为空时取配置 RETRIEVER_TOP_K

        Returns:
            {
                "answer": str,          # AI 回答
                "sources": list,        # 引用来源列表
                "session_id": int,      # 会话 ID
                "message_id": int,      # 消息 ID
                "response_time": float, # 本次耗时（秒）
            }
        """
        start_time = time.time()
        # 请求没指定就落到配置上，避免接口默认值与 .env 里的 RETRIEVER_TOP_K 打架
        top_k = top_k or self.settings.retriever_top_k
        logger.info("收到问题: '{}'（top_k={}）", question[:80], top_k)

        # ---- 步骤 1: 获取或创建会话 ----
        session = self._get_or_create_session(session_id, question)

        # ---- 步骤 2: 从 FAISS 检索相关文本块 ----
        valid_doc_ids = self._get_valid_document_ids()

        # 章节定向：问"创新点/idea"偏向摘要+引言，问"怎么实现"偏向方法章。
        # 只做加权不做硬过滤——章节缺失时硬过滤会直接返回空。
        wanted = set()
        if self.settings.section_aware_retrieval:
            wanted = preferred_sections(question)
        if wanted:
            logger.info(
                "章节定向检索: 意图命中的章节 = {}",
                ", ".join(section_label(s) for s in sorted(wanted)),
            )

        # 候选数：开启重排时多取一些，给 cross-encoder 更大的选择空间；
        # 关闭重排则只要 top_k（省算力，等价于旧行为）
        candidate_k = top_k
        if self.settings.rerank_enabled:
            candidate_k = max(top_k, self.settings.rerank_candidates)

        search_results = self.vector_service.search(
            query=question,
            top_k=candidate_k,
            valid_document_ids=valid_doc_ids,
            preferred_sections=wanted,
        )
        logger.info("向量召回 {} 条候选（目标 top_k={}）", len(search_results), top_k)

        # ---- 步骤 2.5: cross-encoder 重排（精排） ----
        # 双塔向量只看"整体像不像"，cross-encoder 让 query 与 chunk 逐对交互，
        # 能把"提到关键词但不含答案"的段落压下去。失败会自动回退到向量顺序。
        if self.settings.rerank_enabled and len(search_results) > top_k:
            search_results = RerankService.get_instance().rerank(
                query=question,
                candidates=search_results,
                top_k=top_k,
            )
        else:
            search_results = search_results[:top_k]

        logger.info("最终送入上下文的文本块: {} 条", len(search_results))

        # ---- 步骤 3: 构建上下文和来源信息 ----
        context, sources = self._build_context(search_results)

        # ---- 步骤 4: 调用 LLM 生成回答 ----
        answer = self._generate_answer(question, context)

        response_time = round(time.time() - start_time, 2)
        logger.info("回答生成完成，耗时 {}s", response_time)

        # ---- 步骤 5: 保存消息记录 ----
        message = self._save_message(
            session=session,
            question=question,
            answer=answer,
            sources=sources,
            response_time=response_time,
        )

        return {
            "answer": answer,
            "sources": sources,
            "session_id": session.id,
            "message_id": message.id,
            "response_time": response_time,
        }

    # ==========================================================
    # 二、跨文献对比
    # ==========================================================
    def compare(
        self,
        document_ids: list[int],
        question: Optional[str] = None,
        aspect: Optional[str] = None,
        chunks_per_doc: Optional[int] = None,
        session_id: Optional[int] = None,
    ) -> dict:
        """
        跨文献对比：每篇论文独立配额检索 → 章节对齐 → 还原阅读顺序 → 结构化对比。

        与 ask() 的本质区别在检索策略，不在模型调用次数：
        ask() 是"在整个库里找最相关的 k 段"，结果可能全落在同一篇论文里；
        compare() 必须保证**每篇都有配额**，否则短论文在对比表里全是"未提及"。

        Args:
            document_ids: 参与对比的论文 ID（至少 2 篇）
            question: 对比问题（与 aspect 至少提供一个）
            aspect: 对比维度，如"方法""实验设置""创新点"
            chunks_per_doc: 每篇取几段，默认取配置 COMPARE_CHUNKS_PER_DOC
            session_id: 会话 ID（为空则新建）

        Raises:
            ValueError: 论文数量不合法、论文不存在/未处理完成、没有对比依据

        Returns:
            {
                "answer": str,        # 对比结论（表格 + 关键差异）
                "groups": list,       # 每篇论文各自的引用片段
                "stats": dict,        # 片段数 / 上下文字符数 / 是否截断
                "session_id": int|None,
                "message_id": int|None,
                "response_time": float,
            }
        """
        start_time = time.time()

        # ---- 步骤 1: 校验论文 ----
        # dict.fromkeys 去重且保持用户选择顺序（SQL 的 IN 不保证返回顺序）
        ids = list(dict.fromkeys(document_ids))
        if len(ids) < 2:
            raise ValueError("跨文献对比至少需要选择 2 篇论文")
        if len(ids) > self.settings.compare_max_documents:
            raise ValueError(
                f"一次最多对比 {self.settings.compare_max_documents} 篇论文，"
                f"当前选了 {len(ids)} 篇"
            )

        docs = (
            self.db.query(Document)
            .filter(Document.id.in_(ids), Document.status == "completed")
            .all()
        )
        found = {d.id for d in docs}
        missing = [i for i in ids if i not in found]
        if missing:
            raise ValueError(f"论文不存在或尚未处理完成: {missing}")

        doc_map = {d.id: d for d in docs}
        ordered_docs = [doc_map[i] for i in ids]

        # ---- 步骤 2: 决定检索用什么查询、对齐哪个章节 ----
        aspect = (aspect or "").strip()
        question = (question or "").strip()
        if not aspect and not question:
            raise ValueError("请至少填写对比维度或对比问题")

        intent_text = aspect or question
        target_sections = preferred_sections(intent_text)
        if not target_sections:
            # 用户没提到"方法/实验/创新点"这类词时退回默认章节。
            # 默认对齐方法章——"这两篇各自怎么做的"是最常见的对比起点。
            target_sections = {self.settings.compare_default_section}

        retrieval_query = question or aspect
        per_doc = chunks_per_doc or self.settings.compare_chunks_per_doc

        logger.info(
            "开始跨文献对比: {} 篇, 维度='{}', 对齐章节={}, 每篇 {} 段",
            len(ordered_docs), intent_text, sorted(target_sections), per_doc,
        )

        # ---- 步骤 3: 逐篇检索（每篇独立配额） ----
        per_doc_results = [
            {
                "document": doc,
                "chunks": self._compare_chunks_for_doc(
                    document=doc,
                    query=retrieval_query,
                    sections=target_sections,
                    limit=per_doc,
                ),
            }
            for doc in ordered_docs
        ]

        # ---- 步骤 4: 组装上下文（含字符预算截断） ----
        context, groups, stats = self._build_compare_context(
            per_doc_results=per_doc_results,
            aspect=intent_text,
            sections=target_sections,
        )

        if stats["total_chunks"] == 0:
            # 一篇都没检到：不调模型（浪费钱且必然胡编），也不写库
            elapsed = round(time.time() - start_time, 2)
            logger.warning("对比未检索到任何片段，跳过模型调用")
            return {
                "answer": (
                    "选中的论文在目标章节里没有检索到可用文本，暂时无法对比。\n\n"
                    "可以试试：换一个对比维度（例如从「实验」改为「方法」）、"
                    "或确认这些论文是否解析完整。"
                ),
                "groups": groups,
                "stats": {**stats, "elapsed": elapsed},
                "session_id": None,
                "message_id": None,
                "response_time": elapsed,
            }

        # ---- 步骤 5: 调用 LLM 生成对比结论 ----
        answer = self._generate_compare_answer(context)
        response_time = round(time.time() - start_time, 2)

        # ---- 步骤 6: 保存记录（复用会话/消息表，对比记录也能在历史里回看） ----
        session = self._get_or_create_session(
            session_id,
            question or f"对比 {len(ordered_docs)} 篇论文的{aspect}",
        )
        all_sources = [chunk for group in groups for chunk in group["chunks"]]
        message = self._save_message(
            session=session,
            question=question or f"对比 {len(ordered_docs)} 篇论文的{aspect}",
            answer=answer,
            sources=all_sources,
            response_time=response_time,
        )

        stats["elapsed"] = response_time
        logger.info(
            "跨文献对比完成: {} 篇 / {} 段 / 上下文 {} 字符，耗时 {}s",
            len(ordered_docs), stats["total_chunks"], stats["context_chars"], response_time,
        )

        return {
            "answer": answer,
            "groups": groups,
            "stats": stats,
            "session_id": session.id,
            "message_id": message.id,
            "response_time": response_time,
        }

    def _compare_chunks_for_doc(
        self,
        document: Document,
        query: str,
        sections: set[str],
        limit: int,
    ) -> list[dict]:
        """
        为单篇论文检索指定章节的文本块，并还原成阅读顺序。

        这里有整个对比功能最关键的一处修正：
            检索返回的是**相关度顺序**，而对比需要**阅读顺序**。
            问答题里顺序无所谓（模型只需看到相关句子），但对比"方法流程"时，
            如果模型看到的是"第 12 段 → 第 3 段 → 第 8 段"，会把步骤读反。
            所以拿到候选后必须按 chunk_index 升序重排。

        Args:
            document: 目标论文
            query: 检索查询串（对比问题或维度）
            sections: 目标章节集合
            limit: 返回段数上限

        Returns:
            按 chunk_index 升序排列的文本块列表
        """
        # 多取一些候选：章节过滤会淘汰一部分，留出余量
        candidates = self.vector_service.search(
            query=query,
            top_k=max(limit * 3, limit + 4),
            valid_document_ids={document.id},
            preferred_sections=sections,
        )
        if not candidates:
            return []

        # 章节对齐：只保留目标章节的块。
        # 但短论文的某个章节可能只有 1 块，硬过滤会让这一篇几乎为空、
        # 在对比表里全变成"未提及"——所以留了回退：对齐后不足 2 块时退回全部候选，
        # 宁可章节不齐，也不要整篇缺席。
        aligned = [
            c for c in candidates
            if (c.get("section") or "other") in sections
        ]
        if len(aligned) < 2:
            logger.info(
                "《{}》目标章节仅命中 {} 块，回退为不限定章节",
                document.filename, len(aligned),
            )
            aligned = candidates

        aligned.sort(key=lambda c: (c.get("chunk_index") or 0))
        return aligned[:limit]

    def _build_compare_context(
        self,
        per_doc_results: list[dict],
        aspect: str,
        sections: set[str],
    ) -> tuple[str, list[dict], dict]:
        """
        把每篇论文的文本块拼成带编号的对比上下文，并按字符预算截断。

        编号规则：论文用字母 A/B/C，块用 [A-1][A-2]。
        比全局流水号更适合对比场景——模型引用时能同时表达"哪一篇的第几段"。

        预算按篇数均分：如果给某篇多、给某篇少，模型会对某篇"看得更细"，
        对比结论就带上偏差。宁可每篇都截，也不让某一篇独大。

        Returns:
            (context_str, groups, stats)
        """
        settings = self.settings
        letters = "ABCDEFGH"
        doc_count = len(per_doc_results)

        per_doc_budget = max(
            500,
            settings.compare_max_context_chars // max(doc_count, 1),
        )

        section_names = "、".join(
            section_label(s) for s in sorted(sections)
        ) or "未限定"
        header = (
            f"本次对比：{doc_count} 篇论文，对比维度：{aspect}，"
            f"对齐章节：{section_names}。\n"
            f"引用格式为 [论文编号-片段编号]，例如 [A-2] 表示论文A的第2个片段。"
        )

        blocks: list[str] = [header]
        groups: list[dict] = []
        total_chars = len(header)
        truncated: list[str] = []

        for idx, item in enumerate(per_doc_results):
            doc: Document = item["document"]
            letter = letters[idx] if idx < len(letters) else f"P{idx + 1}"

            used = 0
            group_chunks: list[dict] = []
            for chunk in item["chunks"]:
                content = chunk.get("content", "")
                room = per_doc_budget - used
                if room <= 0:
                    truncated.append(doc.filename)
                    break
                # 单块超预算时截断而非整块丢弃——丢掉的可能正好是关键句
                if len(content) > room:
                    content = content[:room] + "…（已截断）"
                    truncated.append(doc.filename)
                used += len(content)

                group_chunks.append({
                    "document_name": doc.filename,
                    "chunk_content": content,
                    "page": chunk.get("page_number"),
                    "section": chunk.get("section"),
                    "relevance_score": chunk.get("relevance_score"),
                    "rerank_score": chunk.get("rerank_score"),
                    "chunk_index": chunk.get("chunk_index"),
                    "ref": f"{letter}-{len(group_chunks) + 1}",
                })

            groups.append({
                "document_id": doc.id,
                "document_name": doc.filename,
                "letter": letter,
                "chunks": group_chunks,
            })

            if not group_chunks:
                blocks.append(
                    f"\n【论文{letter}】{doc.filename}\n（该论文没有检索到可用片段）"
                )
                continue

            lines = [f"\n【论文{letter}】{doc.filename}（{len(group_chunks)} 段）"]
            for i, chunk in enumerate(group_chunks, 1):
                meta_bits = []
                if chunk["page"] is not None:
                    meta_bits.append(f"第 {chunk['page']} 页")
                label = section_label(chunk.get("section"))
                if label:
                    meta_bits.append(label)
                meta = f"（{'，'.join(meta_bits)}）" if meta_bits else ""
                lines.append(f"[{letter}-{i}] {meta}\n{chunk['chunk_content']}")

            block = "\n".join(lines)
            blocks.append(block)
            total_chars += len(block)

        stats = {
            "document_count": doc_count,
            "total_chunks": sum(len(g["chunks"]) for g in groups),
            "context_chars": total_chars,
            "budget_per_doc": per_doc_budget,
            "sections": sorted(sections),
            "truncated_documents": sorted(set(truncated)),
        }
        return "\n".join(blocks), groups, stats

    def _generate_compare_answer(self, context: str) -> str:
        """
        调用 LLM 生成对比结论。

        和问答一样，把异常收敛成一段可读文本而不是向上抛：
        模型调用失败（余额/网络/配额）是可预期的运行时状况，
        变成提示语后前端能直接展示，记录也照样写库便于回溯。
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", COMPARE_SYSTEM_PROMPT),
            ("human", "{instruction}"),
        ])
        chain = prompt | get_llm_client() | StrOutputParser()

        try:
            return chain.invoke({
                "context": context,
                # 指令用固定文案，不拼用户输入：用户问题里若含 { } 会撞花括号模板
                "instruction": "请按上述规则完成对比，先给表格，再给关键差异。",
            })
        except Exception as e:
            logger.error("对比生成失败: {}", str(e))
            return f"抱歉，对比结论生成失败：{str(e)}。请检查 API Key 配置是否正确。"

    # ----------------------------------------------------------
    # 内部方法（问答链路）
    # ----------------------------------------------------------

    def _get_or_create_session(
        self,
        session_id: Optional[int],
        question: str,
    ) -> ChatSession:
        """获取已有会话或创建新会话"""
        if session_id:
            session = self.db.query(ChatSession).filter(
                ChatSession.id == session_id,
                ChatSession.is_active == True,
            ).first()
            if not session:
                raise ValueError(f"会话 {session_id} 不存在或已关闭")
            return session

        # 创建新会话，标题取问题的前 50 个字符
        title = question[:50] + ("..." if len(question) > 50 else "")
        session = ChatSession(title=title)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        logger.info("创建新会话: id={}, title='{}'", session.id, session.title)
        return session

    def _get_valid_document_ids(self) -> set[int]:
        """获取所有已完成处理的文档 ID，用于过滤陈旧向量"""
        docs = self.db.query(Document.id).filter(
            Document.status == "completed"
        ).all()
        return {doc.id for doc in docs}

    def _build_context(
        self,
        search_results: list[dict],
    ) -> tuple[str, list[dict]]:
        """
        将检索结果构建为 LLM 上下文和引用来源列表。

        Args:
            search_results: vector_service.search() 的返回结果

        Returns:
            (context_str, sources_list)
        """
        if not search_results:
            return "未找到相关文档内容。", []

        context_parts = []
        sources = []

        for i, result in enumerate(search_results, 1):
            # 构建上下文片段（带编号，便于 LLM 引用）
            # 注意：i 是"片段编号"，不是文档编号。一篇论文会被切成多个片段，
            # 所以同一个文件可能出现 来源1 来源2 … 这属于正常现象。
            meta_parts = []
            if result.get("page_number") is not None:
                meta_parts.append(f"第{result['page_number']}页")
            sec_label = section_label(result.get("section"))
            if sec_label:
                meta_parts.append(sec_label)
            meta = f"（{'，'.join(meta_parts)}）" if meta_parts else ""

            context_parts.append(
                f"[来源{i}] {result['document_name']}{meta}\n"
                f"{result['content']}"
            )

            # 构建引用来源信息
            sources.append({
                "document_name": result["document_name"],
                "chunk_content": result["content"][:300],  # 截取前300字作为摘要
                "page": result.get("page_number"),
                "section": result.get("section"),
                "relevance_score": result["relevance_score"],
                # 开启重排时才有：0~1 的 cross-encoder 归一化分
                "rerank_score": result.get("rerank_score"),
                "chunk_index": result.get("chunk_index"),
            })

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _generate_answer(self, question: str, context: str) -> str:
        """
        调用 LLM 生成回答。

        使用 LCEL (LangChain Expression Language) 构建 Chain:
        Prompt → LLM → 字符串输出

        说明：这里刻意把异常收敛成一段文字回答而不是向上抛。原因是
        "模型调用失败"属于可预期的运行时状况（余额、网络、配额），把它变成
        一条可读的提示语，前端能直接展示，同时消息记录仍会写库，便于回溯。
        """
        # 构建 Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])

        # 构建 Chain: prompt → llm → 解析为字符串
        chain = prompt | get_llm_client() | StrOutputParser()

        try:
            answer = chain.invoke({
                "context": context,
                "question": question,
            })
            return answer
        except Exception as e:
            logger.error("LLM 调用失败: {}", str(e))
            return f"抱歉，AI 回答生成失败：{str(e)}。请检查 API Key 配置是否正确。"

    def _save_message(
        self,
        session: ChatSession,
        question: str,
        answer: str,
        sources: list[dict],
        response_time: float,
    ) -> ChatMessage:
        """保存聊天消息到数据库"""
        message = ChatMessage(
            session_id=session.id,
            question=question,
            answer=answer,
            source_chunks=sources,
            response_time=response_time,
        )
        self.db.add(message)

        # 更新会话消息计数
        session.message_count += 1

        self.db.commit()
        self.db.refresh(message)

        logger.info(
            "消息已保存: session_id={}, message_id={}, response_time={}s",
            session.id, message.id, response_time,
        )
        return message

    # ----------------------------------------------------------
    # 会话与消息管理
    # 放在 service 层，路由层只负责参数校验和响应组装，不直接写 ORM 查询。
    # ----------------------------------------------------------

    def list_sessions(self, page: int = 1, page_size: int = 20) -> dict:
        """
        分页获取仍处于激活状态的会话（按最后活跃时间倒序）。

        Returns:
            {"total": int, "sessions": list[ChatSession]}
        """
        offset = (page - 1) * page_size
        query = self.db.query(ChatSession).filter(ChatSession.is_active.is_(True))
        total = query.count()
        sessions = (
            query.order_by(ChatSession.updated_at.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )
        return {"total": total, "sessions": sessions}

    def get_messages(self, session_id: int) -> list[ChatMessage]:
        """
        获取指定会话的全部消息（按时间正序）。

        Raises:
            ValueError: 会话不存在
        """
        exists = (
            self.db.query(ChatSession.id)
            .filter(ChatSession.id == session_id)
            .first()
        )
        if not exists:
            raise ValueError(f"会话 {session_id} 不存在")

        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )

    def delete_session(self, session_id: int) -> bool:
        """
        软删除会话：只把 is_active 置为 False，消息记录保留（便于审计/恢复）。

        Returns:
            False 表示会话不存在
        """
        session = (
            self.db.query(ChatSession)
            .filter(ChatSession.id == session_id)
            .first()
        )
        if not session:
            return False

        session.is_active = False
        self.db.commit()
        logger.info("会话已软删除: id={}", session_id)
        return True
