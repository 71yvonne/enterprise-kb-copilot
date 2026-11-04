"""
文件解析模块

根据文件类型选择对应的 LangChain Loader 解析文档内容。
支持三种格式：
- PDF:  使用 PyPDFLoader，逐页解析，保留页码信息
- TXT:  使用 TextLoader，UTF-8 编码（自动回退 GBK）
- MD:   使用 TextLoader，与 TXT 处理方式相同

返回 LangChain 的 Document 对象列表，每个 Document 包含：
- page_content: 文本内容
- metadata: 元数据（如页码、文件来源等）
"""

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from backend.utils.logger import logger


def parse_file(file_path: str, file_type: str) -> list[Document]:
    """
    解析文件内容，返回 LangChain Document 列表。

    Args:
        file_path: 文件在服务器上的完整路径
        file_type: 文件类型（pdf / txt / md）

    Returns:
        list[Document]，每个 Document 包含 page_content 和 metadata

    Raises:
        ValueError: 不支持的文件类型
        Exception: 文件读取或解析失败
    """
    logger.info("开始解析文件: path='{}', type='{}'", file_path, file_type)

    if file_type == "pdf":
        documents = _parse_pdf(file_path)
    elif file_type in ("txt", "md"):
        documents = _parse_text(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {file_type}")

    logger.info(
        "文件解析完成: 共 {} 个文档片段, 总字符数 {}",
        len(documents),
        sum(len(doc.page_content) for doc in documents)
    )

    return documents


def _parse_pdf(file_path: str) -> list[Document]:
    """
    解析 PDF 文件。

    使用 PyPDFLoader 逐页解析，每页生成一个 Document。
    metadata 中包含 page 字段（页码，从 0 开始）。
    """
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # 过滤掉空白页
        documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        logger.info("PDF 解析完成: {} 页（已过滤空白页）", len(documents))
        return documents

    except Exception as e:
        logger.error("PDF 解析失败: {}", str(e))
        raise RuntimeError(f"PDF 文件解析失败: {str(e)}") from e


def _parse_text(file_path: str) -> list[Document]:
    """
    解析文本文件（TXT 或 MD）。

    优先使用 UTF-8 编码，失败则自动回退到 GBK 编码（兼容中文 Windows 文件）。
    整个文件作为一个 Document 返回。
    """
    # 尝试 UTF-8
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        logger.info("文本文件已用 UTF-8 解析成功")
        return documents
    except (UnicodeDecodeError, RuntimeError):
        logger.warning("UTF-8 解析失败，尝试 GBK 编码...")

    # 回退到 GBK
    try:
        loader = TextLoader(file_path, encoding="gbk")
        documents = loader.load()
        logger.info("文本文件已用 GBK 解析成功")
        return documents
    except Exception as e:
        logger.error("文本文件解析失败: {}", str(e))
        raise RuntimeError(f"文本文件解析失败: {str(e)}") from e
