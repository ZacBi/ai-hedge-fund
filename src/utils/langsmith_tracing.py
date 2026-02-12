"""
LangSmith 可观测：LangChain 官方 tracing。
设置 LANGSMITH_TRACING=true 与 LANGSMITH_API_KEY 后，LangChain 会自动上报 trace。
请求结束后调用 langsmith_flush() 确保在响应返回前完成上报。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def is_langsmith_configured() -> bool:
    """是否已启用 LangSmith（用于决定是否 wait_for_all_tracers）。"""
    return (
        os.getenv("LANGSMITH_TRACING", "").lower() in ("true", "1")
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


def langsmith_flush() -> None:
    """请求结束后调用，等待 LangSmith tracer 上报完成。未配置时无操作。"""
    if not is_langsmith_configured():
        return
    try:
        from langchain_core.tracers.langchain import wait_for_all_tracers

        wait_for_all_tracers()
    except Exception as e:
        logger.debug("LangSmith wait_for_all_tracers: %s", e)
