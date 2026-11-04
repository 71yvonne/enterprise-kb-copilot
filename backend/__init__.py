# backend/__init__.py
"""论文阅读助手后端模块

进程环境初始化（**顺序很重要，别挪走**）：
    在导入任何子模块之前，先把 .env 注入 os.environ。

为什么必须放在这里：
    1. `huggingface_hub` 在**模块导入时**就把 HF_HUB_OFFLINE / HF_ENDPOINT
       读成模块级常量，之后再改 os.environ 完全不生效。
    2. `pydantic-settings` 只把 .env 读进 Settings 对象，**不会**导出到进程环境。
    3. 所以注入必须发生在 `backend.services.*`（间接导入 langchain / huggingface_hub）
       被 import 之前。放在 backend 包的 __init__ 里，能保证任意入口
       （python -m backend.main、uvicorn backend.main:app、测试脚本）
       都先走到这里。

不这么做时的症状：
    模型权重明明已经从本地缓存加载成功（日志里有 `Loading weights: 100%`），
    但 huggingface_hub 仍会去 huggingface.co 核对 additional_chat_templates
    这类可选文件；国内网络超时（WinError 10060）后重试 5 次
    （1s/2s/4s/8s/16s），每次启动白等半分钟以上。
"""

from dotenv import load_dotenv

# override=False（默认）：已经在 shell 里设过的同名变量优先，不被 .env 覆盖。
# 这样临时想联网下一模型时，直接在终端 export HF_HUB_OFFLINE=0 即可。
load_dotenv()
