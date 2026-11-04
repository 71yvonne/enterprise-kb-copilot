"""
文档内容清洗

为什么需要：
    PDF 论文末尾的「参考文献 / 致谢 / 附录」对问答几乎没有价值，但它们
    包含大量领域术语（lane detection、dataset、weather…），在向量空间里
    反而和用户问题很接近，会抢走 top-k 的名额。实测 one 篇 10 页论文的
    top-8 里能占到 1–2 条。

处理时机：
    在「切分之后、向量化之前」按块过滤。比在解析阶段按页裁剪更精确，
    因为参考文献标题经常和正文结尾共处同一页。

三条规则（按优先级）：
    1. 参考文献标题（References / Bibliography / 参考文献）：从该行截断，
       并且此后的所有块全部丢弃——论文尾部不可能再有正文。
    2. 致谢标题（Acknowledgement / 致谢）：从该行截断，块内保留正文部分。
    3. 没有标题但内容像参考文献列表（密集的 [12] 标记 + 年份 + et al. 等）
       或者内容过短（页眉页脚残片），直接丢弃。
"""

import re
from dataclasses import dataclass
from typing import Optional

from langchain_core.documents import Document

# ============================================================
# 判定规则
# ============================================================

# 标题必须独立成行，避免匹配正文里的 "see the References section"
_REFERENCE_HEADING = re.compile(
    r"^\s*(?:\d{1,2}[\.\s、]*)?"
    r"(references|reference|bibliography|works\s+cited|参考文献|引用文献)"
    r"\s*[:：]?\s*$",
    re.IGNORECASE,
)

_ACK_HEADING = re.compile(
    r"^\s*(?:\d{1,2}[\.\s、]*)?"
    r"(acknowledge?ments?|acknowledgements?|致谢|致\s*谢)"
    r"\s*[:：]?\s*$",
    re.IGNORECASE,
)

# 参考文献列表的典型特征
_CITATION_MARKER = re.compile(r"\[\d{1,3}\]")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_REFERENCE_HINT = re.compile(
    r"(et\s+al\.?|in\s+proceedings|arxiv|doi|vol\.|pp\.|isbn|http)",
    re.IGNORECASE,
)

# 这些章节无论引用标记多密都必须保留：
# 学术论文的 Introduction / Related Work 段落经常连着引用 3–5 篇文献，
# 用"引用标记数量"判参考文献会把正文误删（真实踩过这个坑）。
_PROTECTED_HEADING = re.compile(
    r"^\s*(?:\d{1,2}[\.\s、]*)?(abstract|introduction|related\s+work|"
    r"method(s|ology)?|experiment(s)?|conclusion(s)?|摘要|引言|绪论|相关工作|方法)\b",
    re.IGNORECASE | re.MULTILINE,
)

# 少于这个字符数的块基本是页眉/页脚/断行残片
_MIN_CHUNK_CHARS = 20
# 截断后剩余正文少于这个长度，整块丢弃（说明这块几乎全是尾部内容）
_MIN_KEEP_PREFIX = 100
# 丢弃比例超过这个值就告警，防止规则误伤整篇
DROP_RATIO_WARN = 0.4


@dataclass
class FilterStats:
    """清洗统计，用于日志"""

    kept: int = 0
    dropped_reference: int = 0
    dropped_ack: int = 0
    dropped_tiny: int = 0
    truncated: int = 0

    @property
    def dropped(self) -> int:
        return self.dropped_reference + self.dropped_ack + self.dropped_tiny


# ============================================================
# 工具函数
# ============================================================

def _cut_at_heading(text: str, pattern: re.Pattern) -> Optional[int]:
    """
    返回第一个匹配标题行的字符偏移；没找到返回 None。

    行首偏移用累计长度计算，不使用 splitlines 的索引，确保截断位置准确。
    """
    offset = 0
    for line in text.splitlines(keepends=True):
        if pattern.match(line):
            return offset
        offset += len(line)
    return None


def looks_like_reference_list(text: str) -> bool:
    """
    没有标题时，靠内容特征判断这块是不是参考文献列表。

    阈值刻意定得高：参考文献条目几乎每行都有 [n]，并且伴随大量年份和
    出版信息；而正文段落即使引用 3–4 篇，也不会有成片的年份。
    """
    markers = len(_CITATION_MARKER.findall(text))
    years = len(_YEAR.findall(text))
    hints = len(_REFERENCE_HINT.findall(text))

    # 典型文献列表：引用标记密集 + 年份成片
    if markers >= 5 and years >= 3:
        return True
    # 没有编号但有出版信息：年份多 + 出版关键词多
    if years >= 4 and hints >= 3:
        return True
    return False


def has_protected_heading(text: str) -> bool:
    """块里出现 Abstract / Introduction / Method 等章节标题时，不做内容判定删除"""
    return bool(_PROTECTED_HEADING.search(text))


def _clone(chunk: Document, text: str) -> Document:
    """保留 metadata，替换正文（原块可能被多页共享，不原地修改）"""
    return Document(page_content=text, metadata=dict(chunk.metadata or {}))


# ============================================================
# 主入口
# ============================================================

def filter_noise_chunks(
    chunks: list[Document],
    enabled: bool = True,
) -> tuple[list[Document], FilterStats]:
    """
    过滤参考文献 / 致谢 / 过短的噪声块。

    Args:
        chunks: 切分后的文本块（按文档顺序）
        enabled: 关闭时原样返回，方便对比实验或排查误伤

    Returns:
        (过滤后的块列表, 统计信息)
    """
    stats = FilterStats()

    if not enabled or not chunks:
        stats.kept = len(chunks)
        return list(chunks), stats

    kept: list[Document] = []
    drop_rest = False  # 见过参考文献标题后，后面全部不要

    for chunk in chunks:
        if drop_rest:
            stats.dropped_reference += 1
            continue

        text = chunk.page_content or ""
        stripped = text.strip()

        if len(stripped) < _MIN_CHUNK_CHARS:
            stats.dropped_tiny += 1
            continue

        # ---- 规则 1：参考文献标题 ----
        cut = _cut_at_heading(text, _REFERENCE_HEADING)
        if cut is not None:
            drop_rest = True
            prefix = text[:cut].strip()
            if len(prefix) < _MIN_KEEP_PREFIX:
                stats.dropped_reference += 1
                continue
            kept.append(_clone(chunk, prefix))
            stats.truncated += 1
            continue

        # ---- 规则 2：致谢标题 ----
        cut = _cut_at_heading(text, _ACK_HEADING)
        if cut is not None:
            prefix = text[:cut].strip()
            if len(prefix) < _MIN_KEEP_PREFIX:
                stats.dropped_ack += 1
                continue
            kept.append(_clone(chunk, prefix))
            stats.truncated += 1
            continue

        # ---- 规则 3：无标题但明显是参考文献列表 ----
        # 有 Abstract / Introduction / Method 这类标题的块一律豁免：
        # 正文段落引用密集是学术写作的常态，不是参考文献。
        if (not has_protected_heading(text)) and looks_like_reference_list(stripped):
            stats.dropped_reference += 1
            continue

        kept.append(chunk)

    stats.kept = len(kept)
    return kept, stats
