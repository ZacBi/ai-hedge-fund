"""
Langfuse 可观测：为 LangChain/LangGraph 提供 callback，用于追踪 LLM 调用。
未配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY 时返回空列表，不影响现有逻辑。
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def is_langfuse_configured() -> bool:
    """是否已配置 Langfuse（用于决定是否 flush）。"""
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def langfuse_flush() -> None:
    """请求结束后调用，确保 trace 在响应返回前上报。未配置时无操作。"""
    if not is_langfuse_configured():
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as e:
        logger.debug("Langfuse flush: %s", e)


def get_langfuse_callbacks(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[Any]:
    """
    若已配置 Langfuse，返回用于 invoke 的 callbacks 列表；否则返回 []。
    用法: graph.invoke(input, config={"callbacks": get_langfuse_callbacks()})
    Langfuse v3 CallbackHandler 不再通过 __init__ 接收 tags/session_id 等，此处保留参数以兼容调用方，创建时使用无参。
    """
    if not is_langfuse_configured():
        return []

    try:
        from langfuse.langchain import CallbackHandler

        # Langfuse v3 CallbackHandler 仅支持无参或少量参数，tags/session_id 等通过 config 传入
        return [CallbackHandler()]
    except Exception as e:
        logger.warning("Langfuse CallbackHandler 不可用，跳过 tracing: %s", e)
        return []
