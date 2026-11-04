"""
章节识别与检索定向

为什么需要：
    naive top-k 检索在"这篇论文的 idea 是什么"这类问题上会失手。全篇几十个
    块里 Abstract/Introduction 通常只占 3–4 块，很容易被 Method/Related Work
    里那些同样高频出现"提出/框架/方法"字样的片段挤出 top-k，模型就给不出答案。

做法：
    1. 入库时：按块顺序扫描章节标题，给每个块打上 section 标签
       （abstract / introduction / related_work / method / experiment /
         conclusion / other），写进 FAISS 元数据。
    2. 查询时：根据问题里的关键词推断"用户想看哪几章"，检索时给这些章节
       的块**加权**而不是硬过滤——硬过滤在章节缺失时会直接返回空结果。

    加权只影响排序，不改变展示给用户的相关度分数，避免分数虚高。
"""

import re
from typing import Optional

SECTION_ABSTRACT = "abstract"
SECTION_INTRODUCTION = "introduction"
SECTION_RELATED = "related_work"
SECTION_METHOD = "method"
SECTION_EXPERIMENT = "experiment"
SECTION_CONCLUSION = "conclusion"
SECTION_OTHER = "other"

# 展示用中文名（前端和日志共用一份定义）
SECTION_LABELS = {
    SECTION_ABSTRACT: "摘要",
    SECTION_INTRODUCTION: "引言",
    SECTION_RELATED: "相关工作",
    SECTION_METHOD: "方法",
    SECTION_EXPERIMENT: "实验",
    SECTION_CONCLUSION: "结论",
    SECTION_OTHER: "正文",
}

# 章节标题词表（英文 + 中文）
_HEADING_KEYWORDS = (
    r"(abstract|introduction|related\s+work|background|preliminar(?:y|ies)|"
    r"method(?:s|ology)?|approach|proposed\s+\w+|experiment(?:s|al)?|evaluation|"
    r"results?|conclusion(?:s)?|discussion|future\s+work|"
    r"摘要|引言|绪论|相关工作|研究背景|方法|实验|结果|结论|讨论)"
)

# 严格式：整行只有编号 + 标题词，可选冒号。
# 例："3. Method" / "REFERENCES" / "摘要"
_HEADING_STRICT = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2})?[\.\s、]*)?" + _HEADING_KEYWORDS + r"\s*[:：]?\s*$",
    re.IGNORECASE,
)

# 宽松式：标题后面紧跟分隔符（"Abstract—We propose..." 这种排版）。
# 只对较短的行启用，避免把正文句子当成标题。
_HEADING_LOOSE = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2})?[\.\s、]*)?" + _HEADING_KEYWORDS + r"\s*[:：.。—–\-]+",
    re.IGNORECASE,
)

# 关键词 → 章节（顺序敏感：越具体的放前面）
_KEYWORD_SECTION = (
    ("abstract", SECTION_ABSTRACT),
    ("摘要", SECTION_ABSTRACT),
    ("introduction", SECTION_INTRODUCTION),
    ("引言", SECTION_INTRODUCTION),
    ("绪论", SECTION_INTRODUCTION),
    ("related", SECTION_RELATED),
    ("background", SECTION_RELATED),
    ("preliminar", SECTION_RELATED),
    ("相关", SECTION_RELATED),
    ("研究背景", SECTION_RELATED),
    ("method", SECTION_METHOD),
    ("approach", SECTION_METHOD),
    ("proposed", SECTION_METHOD),
    ("方法", SECTION_METHOD),
    ("experiment", SECTION_EXPERIMENT),
    ("evaluation", SECTION_EXPERIMENT),
    ("result", SECTION_EXPERIMENT),
    ("实验", SECTION_EXPERIMENT),
    ("结果", SECTION_EXPERIMENT),
    ("conclusion", SECTION_CONCLUSION),
    ("discussion", SECTION_CONCLUSION),
    ("future", SECTION_CONCLUSION),
    ("结论", SECTION_CONCLUSION),
    ("讨论", SECTION_CONCLUSION),
)

# 严格式允许的最大行长度（标题行通常很短）
_MAX_HEADING_LINE = 60
# 宽松式允许的最大行长度（"Abstract—正文…" 会稍长）
_MAX_LOOSE_LINE = 40

# 问题关键词 → 期望章节
_QUERY_SECTION_HINTS = (
    (
        frozenset({SECTION_ABSTRACT, SECTION_INTRODUCTION}),
        (
            "idea", "想法", "创意", "创新", "贡献", "动机", "motivation",
            "为什么", "解决什么问题", "背景", "概述", "主要工作", "摘要",
            "introduction", "abstract", "novel", "这篇论文讲什么", "介绍",
        ),
    ),
    (
        frozenset({SECTION_METHOD}),
        (
            "方法", "怎么实现", "如何实现", "怎么做的", "架构", "结构",
            "流程", "算法", "method", "approach", "framework",
            "architecture", "pipeline", "训练", "网络结构",
        ),
    ),
    (
        frozenset({SECTION_EXPERIMENT}),
        (
            "实验", "结果", "性能", "准确率", "指标", "对比", "baseline",
            "benchmark", "evaluation", "ablation", "消融", "数据集",
        ),
    ),
    (
        frozenset({SECTION_CONCLUSION}),
        ("结论", "总结", "局限", "未来", "conclusion", "limitation", "future work"),
    ),
)


def _map_keyword(keyword: str) -> str:
    lowered = keyword.lower()
    for needle, section in _KEYWORD_SECTION:
        if needle in lowered:
            return section
    return SECTION_OTHER


def detect_section_in_text(text: str) -> Optional[str]:
    """
    在文本里找第一个章节标题，返回对应章节；找不到返回 None。

    先用严格式（整行只有标题词）判断；不中再用宽松式（标题后紧跟分隔符），
    但只对短行启用，避免把 "Results show that…" 这类正文句子当成标题。
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if len(stripped) <= _MAX_HEADING_LINE:
            match = _HEADING_STRICT.match(stripped)
            if match:
                return _map_keyword(match.group(1))

        if len(stripped) <= _MAX_LOOSE_LINE:
            match = _HEADING_LOOSE.match(stripped)
            if match:
                return _map_keyword(match.group(1))

    return None


def annotate_sections(chunks: list) -> list[str]:
    """
    给每个块打章节标签。

    标题只出现在它所在的那个块里，后面的块靠"继承上一个状态"获得章节，
    因此必须按文档顺序遍历。
    """
    current = SECTION_OTHER
    labels: list[str] = []
    for chunk in chunks:
        found = detect_section_in_text(chunk.page_content or "")
        if found is not None:
            current = found
        labels.append(current)
    return labels


def preferred_sections(question: str) -> set[str]:
    """
    根据问题推断用户想看哪几章。

    返回空集合表示"没有明确意图"，此时检索不加权。
    """
    text = (question or "").lower()
    picked: set[str] = set()
    for sections, keywords in _QUERY_SECTION_HINTS:
        for keyword in keywords:
            if keyword in text:
                picked |= sections
                break
    return picked


def section_label(section: str) -> str:
    """章节 key → 中文展示名"""
    return SECTION_LABELS.get(section or SECTION_OTHER, "")
