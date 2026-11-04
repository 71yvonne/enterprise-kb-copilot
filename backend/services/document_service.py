"""
文档处理服务

提供文档的完整生命周期管理：
- CRUD: 创建、查询、列表、删除文档元数据
- 文档处理流水线: 解析 → 切分 → 向量化 → 入库

路由层（routers/documents.py）调用此模块，不直接操作数据库或 AI 组件。
"""

import os
import re
from typing import Optional
from sqlalchemy.orm import Session

from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.models import Document, DocumentChunk
from backend.config import get_settings
from backend.utils.logger import logger
from backend.utils.file_parser import parse_file
from backend.utils.content_filter import filter_noise_chunks, DROP_RATIO_WARN
from backend.utils.section_detector import annotate_sections, section_label, SECTION_OTHER
from backend.services.vector_service import VectorService


# ==============================================================
# 工具函数
# ==============================================================

# Windows 非法字符 + 控制字符
_UNSAFE_FILENAME_CHARS = re.compile(r'[\x00-\x1f<>:"|?*]')
_MAX_STEM_LENGTH = 80


def sanitize_filename(raw_name: str) -> str:
    """
    把客户端提交的文件名收敛成一个安全的纯文件名。

    浏览器通常只发送 basename，但 HTTP 客户端可以提交任意字符串，
    例如 "../../../etc/passwd" 或 "C:\\Windows\\System32\\x.pdf"。
    os.path.join 遇到绝对路径会直接丢弃前缀，因此这里必须先清洗：
    1. 把两种分隔符统一成 "/" 后取最后一段（去掉任何目录成分）
    2. 替换控制字符和 Windows 非法字符
    3. 去掉首尾的点和空格（Windows 不允许以点或空格结尾）
    4. 压缩 ".." 残片，避免相对路径语义残留
    5. 限制主体长度，保留扩展名
    """
    if not raw_name:
        return "document"

    # 1. 统一分隔符，只保留最后一段
    name = str(raw_name).replace("\\", "/").split("/")[-1]

    # 2. 非法字符替换掉
    name = _UNSAFE_FILENAME_CHARS.sub("_", name)

    # 3. 去掉首尾空白与点
    name = name.strip().strip(". ")

    # 4. ".." 残片（防止 name 变成 ".." 或 "a../b" 之类）
    name = name.replace("..", "_")

    if not name:
        return "document"

    # 5. 限制长度，保留扩展名
    stem, ext = os.path.splitext(name)
    if len(stem) > _MAX_STEM_LENGTH:
        stem = stem[:_MAX_STEM_LENGTH]
    return f"{stem}{ext}"


def to_display_page(raw_page) -> Optional[int]:
    """
    把 PyPDFLoader 的 0 起页码转成 1 起的展示页码。

    全链路统一用 1 起页码：数据库 document_chunks.page_number、
    FAISS 元数据、引用卡片、给模型的上下文，都是同一个数字。
    """
    if raw_page is None:
        return None
    try:
        return int(raw_page) + 1
    except (TypeError, ValueError):
        return None


class DocumentService:
    """文档 CRUD + 文档处理流水线"""

    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()

    # ==============================================================
    # 文档处理流水线（核心链路）
    # ==============================================================

    def process_document(self, document_id: int) -> Document:
        """
        文档处理完整流水线：解析 → 切分 → 向量化 → 入库。

        流程图：
            上传文件 → parse_file() 解析为 Document 列表
                     → RecursiveCharacterTextSplitter 切分为小文本块
                     → VectorService.add_texts() 向量化并存入 FAISS
                     → 保存 DocumentChunk 到 MySQL
                     → 更新 Document 状态为 completed

        Args:
            document_id: 要处理的文档 ID（需已创建记录且文件已保存）

        Returns:
            处理完成的 Document 对象

        Raises:
            ValueError: 文档 ID 不存在
            Exception: 处理过程中的任何错误（文档状态会被标记为 failed）
        """
        doc = self.get_document(document_id)
        if not doc:
            raise ValueError(f"文档 id={document_id} 不存在")

        logger.info(
            "===== 开始处理文档: id={}, filename='{}' =====",
            doc.id, doc.filename,
        )

        try:
            # ---- 第 1 步: 解析文件 ----
            logger.info("[1/4] 解析文件...")
            raw_documents = parse_file(doc.file_path, doc.file_type)
            logger.info(
                "[1/4] 解析完成: {} 个文档片段",
                len(raw_documents),
            )

            # ---- 第 2 步: 文本切分 ----
            logger.info("[2/4] 切分文本...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
            )
            chunks = text_splitter.split_documents(raw_documents)
            raw_count = len(chunks)
            logger.info(
                "[2/4] 切分完成: {} 个文本块 (chunk_size={}, overlap={})",
                raw_count, self.settings.chunk_size, self.settings.chunk_overlap,
            )

            if not chunks:
                raise ValueError("文档切分后无有效文本块，请检查文件内容")

            # ---- 第 2.5 步: 内容清洗（去参考文献 / 致谢 / 过短残片） ----
            # 这一步必须在向量化之前：论文尾部的参考文献里全是领域术语，
            # 会挤占 top-k 名额。
            chunks, filter_stats = filter_noise_chunks(
                chunks, enabled=self.settings.filter_references
            )
            if filter_stats.dropped or filter_stats.truncated:
                logger.info(
                    "[2/4] 内容清洗: {} 块 → {} 块（丢弃参考文献 {} / 致谢 {} / 过短 {}，截断 {}）",
                    raw_count, filter_stats.kept,
                    filter_stats.dropped_reference, filter_stats.dropped_ack,
                    filter_stats.dropped_tiny, filter_stats.truncated,
                )
                # 丢弃超过 40% 很可能是规则误伤正文（例如整篇都是引用密集的综述）
                if raw_count > 0 and (filter_stats.dropped / raw_count) > DROP_RATIO_WARN:
                    logger.warning(
                        "内容清洗丢弃了 {:.0%} 的文本块，疑似误伤；"
                        "如确认不需要过滤，可在 .env 设置 FILTER_REFERENCES=false",
                        filter_stats.dropped / raw_count,
                    )

            if not chunks:
                raise ValueError(
                    "文档内容全部被判为参考文献/致谢，未生成任何文本块；"
                    "如需保留请设置 FILTER_REFERENCES=false"
                )

            # ---- 第 3 步: 向量化 + 存入 FAISS ----
            logger.info("[3/4] 向量化并存入 FAISS...")
            texts = [chunk.page_content for chunk in chunks]

            # 章节标签：入库时打标，检索时按问题意图加权
            sections = annotate_sections(chunks)
            section_counts: dict[str, int] = {}
            for name in sections:
                section_counts[name] = section_counts.get(name, 0) + 1
            logger.info(
                "[3/4] 章节分布: {}",
                ", ".join(
                    f"{section_label(k)} {v}"
                    for k, v in sorted(section_counts.items(), key=lambda kv: -kv[1])
                ),
            )

            metadatas = [
                {
                    "document_id": doc.id,
                    "document_name": doc.filename,
                    "chunk_index": i,
                    "page_number": to_display_page(chunk.metadata.get("page")),
                    "section": sections[i] if i < len(sections) else SECTION_OTHER,
                }
                for i, chunk in enumerate(chunks)
            ]

            vector_service = VectorService.get_instance()
            vector_ids = vector_service.add_texts(texts, metadatas)
            logger.info("[3/4] 向量化完成: {} 个向量已入库", len(vector_ids))

            # ---- 第 4 步: 保存文本块到 MySQL ----
            logger.info("[4/4] 保存文本块到数据库...")
            for i, chunk in enumerate(chunks):
                db_chunk = DocumentChunk(
                    document_id=doc.id,
                    chunk_index=i,
                    content=chunk.page_content,
                    page_number=to_display_page(chunk.metadata.get("page")),
                    vector_id=vector_ids[i] if i < len(vector_ids) else None,
                )
                self.db.add(db_chunk)

            # 更新文档状态
            doc.status = "completed"
            doc.chunk_count = len(chunks)
            doc.error_message = None
            self.db.commit()

            logger.info(
                "===== 文档处理完成: id={}, filename='{}', chunks={} =====",
                doc.id, doc.filename, len(chunks),
            )
            return doc

        except Exception as e:
            # 处理失败：标记状态，记录错误信息
            logger.error(
                "文档处理失败: id={}, error='{}'",
                document_id, str(e),
            )

            # 这里必须先 rollback。失败可能发生在 flush 阶段（例如内容超长、
            # 外键冲突），此时 SQLAlchemy 会话已进入"需要回滚"状态，直接
            # commit 会抛 PendingRollbackError:
            #   "This Session's transaction has been rolled back due to a
            #    previous exception during flush."
            # 那个异常会盖掉真正的错误，让排查变成猜谜。
            try:
                self.db.rollback()
            except Exception as rollback_err:
                logger.warning("回滚数据库会话失败: {}", str(rollback_err))

            # 用一个干净的事务写失败状态；本身再失败也只记日志，不掩盖原始异常
            try:
                failed_doc = self.get_document(document_id)
                if failed_doc is not None:
                    failed_doc.status = "failed"
                    failed_doc.error_message = str(e)[:500]
                    self.db.commit()
            except Exception as mark_err:
                logger.error("标记文档为 failed 时出错: {}", str(mark_err))
                try:
                    self.db.rollback()
                except Exception:
                    pass

            raise

    # ==============================================================
    # 基础 CRUD 方法
    # ==============================================================

    def create_document(
        self,
        filename: str,
        file_type: str,
        file_size: int,
        file_path: str,
    ) -> Document:
        """创建文档元数据记录（状态: processing）"""
        doc = Document(
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            file_path=file_path,
            status="processing",
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        logger.info(
            "文档记录已创建: id={}, filename='{}', size={}B",
            doc.id, doc.filename, doc.file_size,
        )
        return doc

    def get_document(self, document_id: int) -> Optional[Document]:
        """根据 ID 查询单个文档"""
        return self.db.query(Document).filter(Document.id == document_id).first()

    def list_documents(self, page: int = 1, page_size: int = 10) -> dict:
        """分页查询文档列表（按创建时间倒序）"""
        offset = (page - 1) * page_size
        total = self.db.query(Document).count()
        documents = (
            self.db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(page_size)
            .all()
        )
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "documents": documents,
        }

    def delete_document(self, document_id: int) -> bool:
        """
        删除文档：本地文件 + MySQL 记录（级联删除文本块）+ FAISS 向量。

        顺序说明：先删数据库再删向量。数据库是事实来源，删完文档后即使
        向量清理失败，检索侧还有 valid_document_ids 兜底，不会返回已删除的
        文档；反之先删向量而数据库失败，就会留下"文档还在但检索不到"的坑。

        向量清理是 best effort：失败只告警，不影响删除结果。

        Returns:
            False 表示文档不存在
        """
        doc = self.db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            return False

        filename = doc.filename

        # 删除本地文件
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
                logger.info("本地文件已删除: {}", doc.file_path)
            except OSError as e:
                logger.error("删除本地文件失败: {}", str(e))

        # 删除数据库记录（级联删除 chunks）
        self.db.delete(doc)
        self.db.commit()
        logger.info("文档已删除: id={}, filename='{}'", document_id, filename)

        # 清理 FAISS 里的向量（懒加载单例：没初始化过就不必为删除去加载模型）
        vector_service = VectorService.peek_instance()
        if vector_service is not None:
            try:
                vector_service.remove_document(document_id)
            except Exception as e:
                logger.warning(
                    "清理文档 {} 的向量失败（可用 /api/v1/documents/rebuild-index 修复）: {}",
                    document_id, str(e),
                )

        return True

    def rebuild_index(self) -> dict:
        """
        用 MySQL 里的文本块重建 FAISS 索引。

        典型用途：
        - 换了嵌入模型（维度变了），不必重新上传文件
        - 索引与数据库漂移时修复

        注意：section 字段没有落库，这里按每个文档的
        块顺序重新推断章节标签，保证重建后的元数据与上传时一致。

        Returns:
            {"vectors": 向量数, "documents": 参与重建的文档数, "chunks": 文本块数}
        """
        from langchain_core.documents import Document as LangchainDocument

        rows = (
            self.db.query(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .filter(Document.status == "completed")
            .order_by(Document.id, DocumentChunk.chunk_index)
            .all()
        )

        vector_service = VectorService.get_instance()

        if not rows:
            vector_service.rebuild([], [])
            return {"vectors": 0, "documents": 0, "chunks": 0}

        # 按文档分组：章节识别依赖块在文档内的先后顺序
        grouped: dict[int, list] = {}
        doc_meta: dict[int, Document] = {}
        for chunk, doc in rows:
            grouped.setdefault(doc.id, []).append(chunk)
            doc_meta[doc.id] = doc

        texts: list[str] = []
        metadatas: list[dict] = []
        for doc_id, chunks in grouped.items():
            doc = doc_meta[doc_id]
            sections = annotate_sections(
                [LangchainDocument(page_content=c.content) for c in chunks]
            )
            for chunk, section in zip(chunks, sections):
                texts.append(chunk.content)
                metadatas.append({
                    "document_id": doc.id,
                    "document_name": doc.filename,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "section": section,
                })

        total = vector_service.rebuild(texts, metadatas)
        logger.info(
            "索引重建完成: {} 个文档 / {} 个块 / {} 个向量",
            len(grouped), len(texts), total,
        )
        return {"vectors": total, "documents": len(grouped), "chunks": len(texts)}

    def save_uploaded_file(self, filename: str, content: bytes) -> str:
        """
        将上传文件保存到磁盘（同名文件自动加序号避免冲突）。

        文件名先经 sanitize_filename 清洗：即使路由层漏了校验，也不会因为
        "../../x.pdf" 或 "C:\\Windows\\x.pdf" 把文件写到 upload_dir 之外。
        """
        os.makedirs(self.settings.upload_dir, exist_ok=True)

        safe_name = sanitize_filename(filename)
        file_path = os.path.join(self.settings.upload_dir, safe_name)

        # 兜底断言：最终路径必须落在 upload_dir 之内
        upload_root = os.path.abspath(self.settings.upload_dir)
        if os.path.commonpath([upload_root, os.path.abspath(file_path)]) != upload_root:
            raise ValueError(f"非法的文件名: {filename!r}")

        if os.path.exists(file_path):
            name, ext = os.path.splitext(safe_name)
            counter = 1
            while os.path.exists(file_path):
                file_path = os.path.join(
                    self.settings.upload_dir,
                    f"{name}_{counter}{ext}",
                )
                counter += 1

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(
            "文件已保存: {} ({} bytes), 原始文件名='{}'",
            file_path, len(content), filename,
        )
        return file_path

    def get_document_chunks(self, document_id: int) -> list[DocumentChunk]:
        """获取某个文档的所有文本块（按序号排序）"""
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )

    def get_statistics(self) -> dict:
        """
        获取论文库统计信息。

        注意 total_vectors 与 total_chunks 可能不一致：删除文档时对向量的清理是 best effort，失败或后端未加载索引时会残留，需调用 /rebuild-index 修复。这里只是如实反映两个数字。
        """
        from sqlalchemy import func

        total_docs = self.db.query(Document).count()
        total_chunks = self.db.query(DocumentChunk).count()

        status_counts = (
            self.db.query(Document.status, func.count(Document.id))
            .group_by(Document.status)
            .all()
        )
        status_map = dict(status_counts)

        # 只读查看单例：索引没被加载过就不该在这里触发模型初始化
        vector_service = VectorService.peek_instance()
        total_vectors = vector_service.total_vectors if vector_service else 0

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "total_vectors": total_vectors,
            "completed": status_map.get("completed", 0),
            "processing": status_map.get("processing", 0),
            "failed": status_map.get("failed", 0),
        }
