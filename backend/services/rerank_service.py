"""
Cross-encoder 重排服务

为什么需要：
    向量检索用的是双塔模型：query 和 chunk 各自独立编码，再算向量距离，
    两者从头到尾没有"互相看过"。cross-encoder 把 (query, chunk) 拼成一条
    输入送进模型，让它们在同一个注意力层里交互，相关性判断明显更准。

    代价是每个候选都要单独做一次前向，所以只能对少量候选（几十条）重排，
    不能直接替代向量检索——正确用法是"向量粗排 + cross-encoder 精排"。

设计：
    - 进程内单例 + 双重检查锁，模型只加载一次
    - 懒加载：启动时不碰这个 1 GB 级模型，首次问答才加载
    - 任何异常（下载失败、内存不足、推理报错）都降级为"保持向量顺序"，
      绝不让重排把整个问答打挂
"""

import math
import threading
import time
from typing import Optional

from backend.config import get_settings
from backend.utils.logger import logger


def _sigmoid(value: float) -> float:
    """
    把 cross-encoder 的 logits 映射到 0~1，便于展示。

    bge-reranker 输出的是二分类 logits，原始值可能是负数或大于 1，
    直接当百分比显示会让人困惑。排序用的是原始 logits（单调等价），
    这里只影响展示。
    """
    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


class RerankService:
    """Cross-encoder 重排（线程安全的单例）"""

    _instance: Optional["RerankService"] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "RerankService":
        """获取单例；首次调用不会加载模型（交给 rerank 时懒加载）"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = RerankService()
        return cls._instance

    @classmethod
    def peek_instance(cls) -> Optional["RerankService"]:
        """只读查看单例状态，不触发创建"""
        return cls._instance

    def __init__(self):
        settings = get_settings()
        self.model_name = settings.rerank_model
        self._model = None
        self._load_failed = False
        self._load_lock = threading.Lock()

    # ----------------------------------------------------------
    # 模型加载
    # ----------------------------------------------------------
    def _load_model(self):
        """
        懒加载 cross-encoder 模型。

        返回 None 表示加载失败（已记录日志），调用方应回退到向量顺序。
        失败只记一次，避免每次请求都重复报错。
        """
        if self._model is not None or self._load_failed:
            return self._model

        with self._load_lock:
            if self._model is not None or self._load_failed:
                return self._model

            try:
                # 延迟导入：没开重排时完全不碰 sentence-transformers
                from sentence_transformers import CrossEncoder

                started = time.time()
                self._model = CrossEncoder(self.model_name, device="cpu", max_length=512)
                logger.info(
                    "重排模型已加载: {}（耗时 {:.1f}s）",
                    self.model_name, time.time() - started,
                )
            except Exception as e:
                self._load_failed = True
                logger.warning(
                    "重排模型加载失败，后续问答将跳过重排（可用 RERANK_ENABLED=false 关闭）: {}",
                    str(e),
                )

        return self._model

    # ----------------------------------------------------------
    # 重排
    # ----------------------------------------------------------
    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int,
        min_score: Optional[float] = None,
    ) -> list[dict]:
        """
        用 cross-encoder 对候选重新打分排序。

        Args:
            query: 用户问题
            candidates: 向量检索返回的候选（每项至少含 "content"）
            top_k: 返回条数
            min_score: 可选的最低分阈值（logits，未校准，默认不过滤）

        Returns:
            重排后的前 top_k 条；模型不可用或推理失败时原样返回前 top_k 条。
            每条结果会带上 `rerank_score`（0~1，仅用于展示；排序用原始 logits）。
        """
        if not candidates:
            return []

        model = self._load_model()
        if model is None:
            return candidates[:top_k]

        try:
            started = time.time()
            pairs = [(query, c.get("content", "")) for c in candidates]
            scores = model.predict(pairs, batch_size=16, show_progress_bar=False)

            ranked = sorted(
                zip(candidates, (float(s) for s in scores)),
                key=lambda item: item[1],
                reverse=True,
            )

            if min_score is not None:
                ranked = [item for item in ranked if item[1] >= min_score]

            top = ranked[:top_k]

            # 把归一化后的分数挂回结果，供前端展示"重排 xx%"
            for candidate, raw_score in top:
                candidate["rerank_score"] = round(_sigmoid(raw_score), 4)

            if top:
                logger.info(
                    "重排完成: {} 条候选 → {} 条，最高分 {:.3f}，最低分 {:.3f}（耗时 {:.2f}s）",
                    len(candidates), len(top), top[0][1], top[-1][1],
                    time.time() - started,
                )
            else:
                logger.warning("重排后没有候选通过阈值，返回空结果")

            return [item[0] for item in top]

        except Exception as e:
            logger.warning("重排推理失败，回退到向量顺序: {}", str(e))
            return candidates[:top_k]
