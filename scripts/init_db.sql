-- ============================================================
-- 论文阅读助手 — 数据库初始化脚本
-- 由 docker-compose 中的 MySQL 容器启动时自动执行
-- 也可手动执行: mysql -u root -p < scripts/init_db.sql
--
-- 4 张表：documents / document_chunks / chat_sessions / chat_messages
-- 注意：章节标签（section）只存在于 FAISS 元数据，不落库
-- ============================================================

CREATE DATABASE IF NOT EXISTS kb_copilot
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE kb_copilot;

-- ==================== 1. 文档表 ====================
-- 记录用户上传的每一个文档的基本信息和处理状态
CREATE TABLE IF NOT EXISTS documents (
    id              INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    filename        VARCHAR(255)    NOT NULL COMMENT '原始文件名',
    file_type       VARCHAR(10)     NOT NULL COMMENT '文件类型: pdf/txt/md',
    file_size       INT             NOT NULL COMMENT '文件大小(字节)',
    file_path       VARCHAR(500)    NOT NULL COMMENT '服务器存储路径',
    chunk_count     INT             DEFAULT 0 COMMENT '切分后的文本块数量',
    status          VARCHAR(20)     DEFAULT 'processing' COMMENT '处理状态: processing/completed/failed',
    error_message   VARCHAR(500)    DEFAULT NULL COMMENT '处理失败时的错误信息',
    created_at      DATETIME        DEFAULT CURRENT_TIMESTAMP COMMENT '上传时间',
    updated_at      DATETIME        DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_doc_status (status),
    INDEX idx_doc_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='文档元数据表';

-- ==================== 2. 文档文本块表 ====================
-- 一个文档被切分后得到的文本块，每个块对应 FAISS 中的一个向量
CREATE TABLE IF NOT EXISTS document_chunks (
    id              INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    document_id     INT             NOT NULL COMMENT '所属文档 ID',
    chunk_index     INT             NOT NULL COMMENT '块在文档中的序号(从0开始)',
    content         TEXT            NOT NULL COMMENT '文本块内容',
    page_number     INT             DEFAULT NULL COMMENT '所在页码(仅PDF有效)',
    vector_id       VARCHAR(100)    DEFAULT NULL COMMENT '在FAISS向量库中的ID',
    created_at      DATETIME        DEFAULT CURRENT_TIMESTAMP COMMENT '入库时间',
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    INDEX idx_chunk_document_id (document_id),
    INDEX idx_chunk_vector_id (vector_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='文档文本块表';

-- ==================== 3. 聊天会话表 ====================
-- 一个会话代表一次完整的对话（类似 ChatGPT 左侧的对话列表）
CREATE TABLE IF NOT EXISTS chat_sessions (
    id              INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    title           VARCHAR(255)    DEFAULT '新对话' COMMENT '会话标题',
    is_active       TINYINT(1)      DEFAULT 1 COMMENT '是否激活(0=软删除)',
    message_count   INT             DEFAULT 0 COMMENT '消息数量',
    created_at      DATETIME        DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at      DATETIME        DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '最后活跃时间',
    INDEX idx_session_active (is_active),
    INDEX idx_session_updated (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='聊天会话表';

-- ==================== 4. 聊天消息表 ====================
-- 一条记录 = 一问一答（用户问题 + AI 回答 + 引用来源）
CREATE TABLE IF NOT EXISTS chat_messages (
    id              INT AUTO_INCREMENT PRIMARY KEY COMMENT '自增主键',
    session_id      INT             NOT NULL COMMENT '所属会话 ID',
    question        TEXT            NOT NULL COMMENT '用户提问',
    answer          TEXT            NOT NULL COMMENT 'AI 回答',
    source_chunks   JSON            DEFAULT NULL COMMENT '引用来源(JSON数组)',
    response_time   FLOAT           DEFAULT NULL COMMENT 'AI响应耗时(秒)',
    created_at      DATETIME        DEFAULT CURRENT_TIMESTAMP COMMENT '消息时间',
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    INDEX idx_msg_session_id (session_id),
    INDEX idx_msg_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='聊天消息表';
