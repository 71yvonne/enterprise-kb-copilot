"""
文档管理路由

提供文档的上传、列表查询、详情查询、删除、统计等 API 接口。
上传接口会触发完整的文档处理流水线：解析 → 切分 → 清洗 → 向量化 → 入库。

关于同步/异步：
    本模块的路由**全部使用同步 def**。上传解析、切分、embedding、FAISS 写入
    都是阻塞调用，如果写成 async def，这些耗时会直接卡住事件循环，导致同一
    进程内其它请求（包括健康检查）全部排队。写成 def 后 FastAPI 会把它们丢到
    线程池执行，互不影响。
"""

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from backend.config import get_settings
from backend.database import get_db
from backend.schemas import (
    BaseResponse,
    DocumentDetailResponse,
    DocumentInfo,
    DocumentListData,
    DocumentListResponse,
    DocumentStatsResponse,
    DocumentUploadResponse,
    RebuildIndexResponse,
)
from backend.services.document_service import DocumentService, sanitize_filename
from backend.utils.logger import logger

router = APIRouter(prefix="/api/v1/documents", tags=["文档管理"])

# 允许上传的文件类型（只做粗筛，真正的解析由 utils/file_parser.py 负责）
ALLOWED_EXTENSIONS = ("pdf", "txt", "md")


def _get_file_extension(filename: str) -> str:
    """提取文件扩展名（小写，不含点）；没有扩展名时返回空串"""
    if not filename or "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


# ----------------------------------------------------------
# POST /upload — 上传并处理文档
# ----------------------------------------------------------
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="上传文档",
    description="上传 PDF/TXT/MD 文件，自动执行: 解析 → 切分 → 向量化 → 入库。"
)
def upload_document(
    file: UploadFile = File(..., description="支持 .pdf / .txt / .md 文件"),
    db: Session = Depends(get_db),
):
    settings = get_settings()

    # 1. 清洗文件名（防路径穿越：客户端可以提交 "../../x.pdf" 或 "C:\\x\\y.pdf"）
    safe_name = sanitize_filename(file.filename)

    # 2. 校验文件类型（按清洗后的名字取扩展名）
    file_ext = _get_file_extension(safe_name)
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 '{file_ext}'，仅支持: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # 3. 校验大小：先看声明的长度，避免把超大文件整个读进内存
    max_bytes = settings.max_upload_bytes
    declared_size = getattr(file, "size", None) or 0
    if max_bytes and declared_size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过大小上限 {settings.max_upload_mb} MB",
        )

    # 4. 读取内容（同步读取：本路由已运行在线程池中）
    file_content = file.file.read()
    file_size = len(file_content)
    if file_size == 0:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if max_bytes and file_size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过大小上限 {settings.max_upload_mb} MB",
        )

    logger.info(
        "开始处理上传: original='{}', stored='{}', size={}B",
        file.filename, safe_name, file_size,
    )

    service = DocumentService(db)

    try:
        # 5. 落盘（内部会再清洗一次文件名，双重保险）
        file_path = service.save_uploaded_file(safe_name, file_content)

        # 6. 建记录（status=processing），随后跑完整流水线
        doc = service.create_document(
            filename=safe_name,
            file_type=file_ext,
            file_size=file_size,
            file_path=file_path,
        )
        doc = service.process_document(doc.id)
        db.refresh(doc)

        return DocumentUploadResponse(
            code=200,
            message=f"文档上传并处理成功，共生成 {doc.chunk_count} 个文本块",
            data=DocumentInfo.model_validate(doc),
        )

    except Exception as e:
        # 处理失败时 process_document 已经把文档状态写成 failed 并记录了原因，
        # 这里只负责把错误回给调用方；用 exception 保留堆栈便于排查。
        logger.exception("文档上传处理失败: {}", str(e))
        raise HTTPException(status_code=500, detail=f"文档处理失败: {str(e)}") from e


# ----------------------------------------------------------
# GET / — 文档列表
# ----------------------------------------------------------
@router.get(
    "",
    response_model=DocumentListResponse,
    summary="获取文档列表",
    description="分页查询已上传的文档列表，按上传时间倒序排列。"
)
def list_documents(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(10, ge=1, le=100, description="每页条数，最多 100"),
    db: Session = Depends(get_db),
):
    service = DocumentService(db)
    result = service.list_documents(page=page, page_size=page_size)

    return DocumentListResponse(
        code=200,
        data=DocumentListData(
            total=result["total"],
            page=result["page"],
            page_size=result["page_size"],
            documents=[DocumentInfo.model_validate(doc) for doc in result["documents"]],
        ),
    )


# ----------------------------------------------------------
# GET /stats/overview — 文档统计
# 注意：必须声明在 /{document_id} 之前，否则 "stats" 会被当成 document_id
# ----------------------------------------------------------
@router.get(
    "/stats/overview",
    response_model=DocumentStatsResponse,
    summary="文档统计",
    description="获取论文库的文档统计信息。"
)
def get_document_stats(db: Session = Depends(get_db)):
    service = DocumentService(db)
    return DocumentStatsResponse(code=200, data=service.get_statistics())


# ----------------------------------------------------------
# POST /rebuild-index — 用数据库内容重建向量索引
# 注意：必须声明在 /{document_id} 之前，否则会被当成 document_id 解析
# ----------------------------------------------------------
@router.post(
    "/rebuild-index",
    response_model=RebuildIndexResponse,
    summary="重建向量索引",
    description=(
        "用 MySQL 中所有已完成文档的文本块重新生成 FAISS 索引。"
        "适用场景：切换嵌入模型、索引与数据库漂移。"
        "会对每个块重新做 embedding，耗时与块数成正比。"
    ),
)
def rebuild_index(db: Session = Depends(get_db)):
    result = DocumentService(db).rebuild_index()
    return RebuildIndexResponse(
        code=200,
        message=f"索引已重建，共 {result['vectors']} 个向量",
        data=result,
    )


# ----------------------------------------------------------
# GET /{document_id} — 文档详情
# ----------------------------------------------------------
@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
    summary="获取文档详情",
)
def get_document(document_id: int, db: Session = Depends(get_db)):
    service = DocumentService(db)
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentDetailResponse(code=200, data=DocumentInfo.model_validate(doc))


# ----------------------------------------------------------
# DELETE /{document_id} — 删除文档
# ----------------------------------------------------------
@router.delete(
    "/{document_id}",
    response_model=BaseResponse,
    summary="删除文档",
    description="删除指定文档及其关联的文本块和本地文件。"
)
def delete_document(document_id: int, db: Session = Depends(get_db)):
    service = DocumentService(db)
    if not service.delete_document(document_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    return BaseResponse(code=200, message="文档已删除")
