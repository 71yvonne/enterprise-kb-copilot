# ============================================================
# 论文阅读助手 — Dockerfile
#
# 构建统一镜像，通过 docker-compose 中的 command 字段
# 分别启动 FastAPI 后端和 Streamlit 前端。
#
# 构建命令:  docker build -t paperlens .
# 镜像基础:  python:3.11-slim (~150MB)
# ============================================================

FROM python:3.11-slim AS base

# ---- 元信息 ----
LABEL maintainer="your-email@example.com"
LABEL description="论文阅读助手 - 章节感知检索与跨文献对比"

# ---- 环境变量 ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# ---- 工作目录 ----
WORKDIR /app

# ---- 系统依赖 ----
# gcc/libffi-dev: 编译部分 Python C 扩展（cryptography 等）
# curl: 后端健康检查（docker-compose 里的 healthcheck 会用到）
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libffi-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖（单独 COPY，利用 Docker 层缓存） ----
COPY requirements.txt .
RUN pip install -r requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
    && rm -rf /root/.cache/pip

# ---- 项目代码 ----
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/

# ---- 创建运行时目录 ----
RUN mkdir -p data/uploads data/vector_store logs

# ---- 暴露端口 ----
# 8000: FastAPI 后端
# 8501: Streamlit 前端
EXPOSE 8000 8501

# 健康检查由 docker-compose.yml 中的各服务单独配置（后端检查 8000，前端不需要）

# ---- 默认启动后端（docker-compose 中会覆盖此命令） ----
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
