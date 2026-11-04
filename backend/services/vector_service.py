"""
FAISS 向量服务

管理 FAISS 向量库的完整生命周期：
- 初始化 / 加载已有索引
- 添加文档向量
- 删除某个文档的全部向量
- 用数据库内容整体重建索引
- 相似度检索
- 持久化到磁盘

使用单例模式，确保整个应用共享同一个 FAISS 实例。
通过线程锁保证并发安全。

设计说明（删除向量）：
    FAISS 的 IndexFlatL2 支持 remove_ids，但它会**前移后续向量**，而
    LangChain 维护的 index_to_docstore_id（位置 → docstore id）不会跟着变。
    直接删会让"位置"和"文本"整体错位，检索会返回别人的内容。
    因此这里在删除后按保留顺序重建 docstore 与位置映射（见 remove_document）。
"""

import os
import uuid
import threading
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from backend.config import get_settings
from backend.utils.logger import logger


class VectorService:
    """FAISS 向量库管理（线程安全的单例）"""

    _instance: Optional["VectorService"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "VectorService":
        """
        获取 VectorService 单例。

        第一次调用时初始化 Embedding 模型和加载已有索引（本地模型首次加载需要
        数秒），后续调用直接返回已有实例。
        """
        if cls._instance is None:
            with cls._lock:
                # 双重检查锁定，防止并发时重复创建
                if cls._instance is None:
                    cls._instance = VectorService()
        return cls._instance

    @classmethod
    def peek_instance(cls) -> Optional["VectorService"]:
        """
        只读地查看单例是否已初始化，**不会触发创建**。

        用于健康检查 / 统计接口：如果索引还没被任何请求加载过，就不应该在
        这里付出几秒的模型初始化代价。
        """
        return cls._instance

    def __init__(self):
        """初始化 Embedding 模型和 FAISS 索引"""
        settings = get_settings()

        # ---- Embedding 模型：两种后端，由 EMBEDDING_PROVIDER 选择 ----
        provider = (settings.embedding_provider or "openai").strip().lower()
        if provider in ("huggingface", "hf", "local"):
            # 本地推理：不消耗 API 额度，首次使用会从 HuggingFace 下载模型
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            except ImportError as exc:
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=huggingface 需要在 venv 中安装 sentence-transformers，"
                    "执行: venv\\Scripts\\python.exe -m pip install sentence-transformers"
                ) from exc

            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model_name,
                model_kwargs={"device": "cpu"},
                # 归一化后 L2 距离 ≈ 余弦距离，检索分数更稳定
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(
                "Embedding 模型已初始化: provider=huggingface, model='{}'（本地推理）",
                settings.embedding_model_name,
            )
        else:
            self.embeddings = OpenAIEmbeddings(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model=settings.embedding_model_name,
            )
            logger.info(
                "Embedding 模型已初始化: provider=openai, model='{}', base_url='{}'",
                settings.embedding_model_name, settings.openai_base_url,
            )

        # FAISS 相关配置
        self.store_dir = settings.vector_store_dir
        self.vector_store: Optional[FAISS] = None
        self._write_lock = threading.Lock()

        # 尝试加载已有索引
        self._load_index()

    # ----------------------------------------------------------
    # 加载已有索引
    # ----------------------------------------------------------
    def _load_index(self) -> None:
        """从磁盘加载已有 FAISS 索引（如果存在）"""
        index_path = os.path.join(self.store_dir, "index.faiss")
        if os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.store_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,  # 信任本地数据
                )
                doc_count = self.vector_store.index.ntotal
                logger.info("FAISS 索引已从磁盘加载: {} 个向量", doc_count)
            except Exception as e:
                logger.error("FAISS 索引加载失败: {}，将创建新索引", str(e))
                self.vector_store = None
        else:
            logger.info("未找到已有 FAISS 索引，等待首次文档上传后创建")

    # ----------------------------------------------------------
    # 保存索引到磁盘
    # ----------------------------------------------------------
    def _save_index(self) -> None:
        """将 FAISS 索引持久化到磁盘"""
        if self.vector_store is not None:
            os.makedirs(self.store_dir, exist_ok=True)
            self.vector_store.save_local(self.store_dir)
            logger.info("FAISS 索引已保存到: {}", self.store_dir)

    # ----------------------------------------------------------
    # 添加文档向量
    # ----------------------------------------------------------
    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict],
    ) -> list[str]:
        """
        将文本列表向量化并添加到 FAISS 索引。

        Args:
            texts: 文本内容列表
            metadatas: 每条文本对应的元数据列表，结构示例:
                {
                    "document_id": 1,
                    "document_name": "attention-is-all-you-need.pdf",
                    "chunk_index": 0,
                    "page_number": 1
                }

        Returns:
            向量 ID 列表（UUID 字符串），用于存入 DocumentChunk.vector_id
        """
        if not texts:
            logger.warning("add_texts 收到空文本列表，跳过")
            return []

        # 为每条文本生成唯一 ID
        ids = [str(uuid.uuid4()) for _ in texts]

        logger.info(
            "正在向量化 {} 条文本（来自文档: '{}'）...",
            len(texts),
            metadatas[0].get("document_name", "未知") if metadatas else "未知",
        )

        with self._write_lock:
            if self.vector_store is None:
                # 首次创建索引
                self.vector_store = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info("FAISS 索引已创建，首批 {} 个向量", len(texts))
            else:
                # 追加到已有索引
                self.vector_store.add_texts(
                    texts=texts,
                    metadatas=metadatas,
                    ids=ids,
                )
                logger.info(
                    "已追加 {} 个向量，索引总计 {} 个向量",
                    len(texts), self.vector_store.index.ntotal,
                )

            # 持久化
            self._save_index()

        return ids

    # ----------------------------------------------------------
    # 删除某个文档的全部向量
    # ----------------------------------------------------------
    def remove_document(self, document_id: int) -> int:
        """
        删除指定文档在索引中的全部向量，并修复位置映射。

        实现要点：
            1. 先按"位置 → docstore id"取出该文档占用的位置
            2. index.remove_ids 删除这些位置（FAISS 会把后面的向量前移）
            3. 用保留顺序重建 docstore 与 index_to_docstore_id

        第 3 步是关键：跳过后，LangChain 的 position → id 映射就会错位，
        之后任何检索都会把 A 的位置映射到 B 的文本上。

        Args:
            document_id: 文档 ID

        Returns:
            实际删除的向量数量（0 表示没有该文档的向量或索引不可用）
        """
        with self._write_lock:
            store = self.vector_store
            if store is None or store.index is None or store.index.ntotal == 0:
                return 0

            positions = sorted(store.index_to_docstore_id)
            if len(positions) != store.index.ntotal:
                # 映射与索引数量不一致说明索引已损坏，定向删除不可靠，
                # 交给 rebuild_index() 从数据库重建。
                logger.warning(
                    "索引与位置映射不一致（映射 {} 条 / 索引 {} 条），跳过定向删除",
                    len(positions), store.index.ntotal,
                )
                return 0

            keep_ids: list[str] = []
            keep_docs: list[Document] = []
            drop_positions: list[int] = []

            for pos in positions:
                docstore_id = store.index_to_docstore_id[pos]
                # InMemoryDocstore.search 在 id 缺失时会抛 ValueError（不是返回 None），
                # 这里统一收敛成 None，按"孤儿向量"处理
                try:
                    doc = store.docstore.search(docstore_id)
                except Exception:
                    doc = None

                if doc is not None and doc.metadata.get("document_id") == document_id:
                    drop_positions.append(pos)
                elif doc is not None:
                    keep_ids.append(docstore_id)
                    keep_docs.append(doc)
                else:
                    # 位置存在但 docstore 里找不到文本：孤儿向量，一并丢弃
                    drop_positions.append(pos)

            if not drop_positions:
                return 0

            import numpy as np  # 局部导入：只有删除/重建时才需要 numpy
            from langchain_community.docstore.in_memory import InMemoryDocstore

            store.index.remove_ids(np.array(drop_positions, dtype="int64"))

            # 重建 docstore 与位置映射，保证 0..n-1 连续且与文本一一对应
            store.docstore = InMemoryDocstore(dict(zip(keep_ids, keep_docs)))
            store.index_to_docstore_id = {
                i: docstore_id for i, docstore_id in enumerate(keep_ids)
            }

            self._save_index()

            removed = len(drop_positions)
            logger.info(
                "已删除文档 id={} 的 {} 个向量，索引剩余 {} 个",
                document_id, removed, store.index.ntotal,
            )
            return removed

    # ----------------------------------------------------------
    # 用数据库内容整体重建索引
    # ----------------------------------------------------------
    def rebuild(self, texts: list[str], metadatas: list[dict]) -> int:
        """
        丢弃旧索引，用给定文本重建（会覆盖磁盘上的索引文件）。

        用途：
        - 切换嵌入模型后不必重新上传文件，直接从 MySQL 的文本块重建
        - 索引与数据库出现漂移时修复

        注意：会对每个块重新做一次 embedding，耗时与块数成正比。

        Args:
            texts: 全部文本块内容
            metadatas: 与 texts 一一对应的元数据

        Returns:
            重建后的向量总数
        """
        with self._write_lock:
            if not texts:
                self.vector_store = None
                for filename in ("index.faiss", "index.pkl"):
                    path = os.path.join(self.store_dir, filename)
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                        except OSError as e:
                            logger.warning("删除旧索引文件失败 {}: {}", path, str(e))
                logger.info("索引已清空（没有任何文本块）")
                return 0

            ids = [str(uuid.uuid4()) for _ in texts]
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids,
            )
            self._save_index()
            total = self.vector_store.index.ntotal
            logger.info("索引已重建：{} 个向量", total)
            return total

    # ----------------------------------------------------------
    # 相似度检索
    # ----------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 4,
        valid_document_ids: Optional[set[int]] = None,
        preferred_sections: Optional[set[str]] = None,
        section_boost: float = 0.25,
    ) -> list[dict]:
        """
        根据查询文本进行相似度检索。

        Args:
            query: 用户查询文本
            top_k: 返回最相关的 k 条结果
            valid_document_ids: 有效文档 ID 集合（用于过滤已删除文档的陈旧向量）
            preferred_sections: 期望章节集合（如 {"abstract", "introduction"}）。
                命中这些章节的块会在**排序**上加分，但展示给用户的相关度分数不变，
                避免分数虚高。
            section_boost: 章节加权值，只影响排序

        Returns:
            检索结果列表，每项包含:
            {
                "content": str,          # 文本块内容
                "document_name": str,    # 来源文档名
                "document_id": int,      # 文档 ID
                "chunk_index": int,      # 块序号
                "page_number": int|None, # 页码
                "section": str,          # 章节
                "relevance_score": float # 相关度分数 (0~1, 越大越相关)
            }
        """
        # 索引未加载或一条向量都没有时直接返回：FAISS 对 k=0 的行为不可靠
        if self.vector_store is None or self.total_vectors == 0:
            logger.warning("FAISS 索引为空，无法检索")
            return []

        preferred = set(preferred_sections or ())

        # 多取一些结果：要过滤陈旧向量，章节加权还需要更大的候选池
        fetch_k = top_k * 3 if valid_document_ids is not None else top_k
        if preferred:
            fetch_k = max(fetch_k, top_k * 4)

        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=min(fetch_k, self.vector_store.index.ntotal),  # 不超过总数
            )
        except Exception as e:
            logger.error("FAISS 检索失败: {}", str(e))
            return []

        # 格式化结果 + 章节加权（仅用于排序）
        ranked = []
        for doc, distance in results:
            doc_id = doc.metadata.get("document_id")

            # 过滤已删除文档的陈旧向量
            if valid_document_ids is not None and doc_id not in valid_document_ids:
                continue

            section = doc.metadata.get("section") or "other"
            # L2距离 → 相关度分数：1/(1+d)，越大越相关
            base_score = 1.0 / (1.0 + float(distance))
            rank_score = base_score + (section_boost if section in preferred else 0.0)

            ranked.append((
                rank_score,
                {
                    "content": doc.page_content,
                    "document_name": doc.metadata.get("document_name", "未知"),
                    "document_id": doc_id,
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "page_number": doc.metadata.get("page_number"),
                    "section": section,
                    "relevance_score": round(base_score, 4),
                },
            ))

        # 有章节偏好时按加权分数重排；否则保持 FAISS 的距离顺序
        if preferred:
            ranked.sort(key=lambda item: item[0], reverse=True)

        formatted = [item[1] for item in ranked[:top_k]]

        logger.info(
            "检索完成: query='{}...', 返回 {} 条结果{}",
            query[:30], len(formatted),
            f", 章节加权={sorted(preferred)}" if preferred else "",
        )
        return formatted

    # ----------------------------------------------------------
    # 索引状态
    # ----------------------------------------------------------
    @property
    def total_vectors(self) -> int:
        """当前索引中的向量总数"""
        if self.vector_store is None:
            return 0
        return self.vector_store.index.ntotal

    @property
    def is_ready(self) -> bool:
        """索引是否可用"""
        return self.vector_store is not None and self.vector_store.index.ntotal > 0
