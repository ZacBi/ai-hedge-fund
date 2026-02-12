"""
Load prompt template by name: from Langfuse when configured, else from local registry.
Returns a LangChain ChatPromptTemplate; callers invoke it with their variables and pass
the result to call_llm(prompt=..., ...).
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from src.prompts.registry import get_default_messages
from src.utils.langfuse_callback import is_langfuse_configured

logger = logging.getLogger(__name__)


def get_prompt_template(
    name: str,
    label: str = "production",
    **compile_kwargs: Any,
) -> ChatPromptTemplate:
    """
    Return a ChatPromptTemplate for the given prompt name.

    When Langfuse is configured (LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY), fetches
    the prompt from Langfuse (type=chat, with optional label/version). On missing
    config or any error, falls back to the local registry default.

    Args:
        name: Prompt name (e.g. "hedge-fund/ben_graham"), must match registry and Langfuse.
        label: Langfuse label (e.g. "production") when fetching from Langfuse.
        **compile_kwargs: Unused; reserved for future use (e.g. pre-invoke).

    Returns:
        ChatPromptTemplate. Call template.invoke({...}) with your variables, then pass
        the result to call_llm(prompt=..., ...).
    """
    _ = compile_kwargs  # reserved
    if is_langfuse_configured():
        try:
            from langfuse import get_client

            client = get_client()
            pf = client.get_prompt(name, label=label)
            lc_prompt = pf.get_langchain_prompt()
            # get_langchain_prompt() for chat returns a list of message-like items
            if isinstance(lc_prompt, list):
                return ChatPromptTemplate.from_messages(lc_prompt)
            return ChatPromptTemplate.from_template(lc_prompt)
        except Exception as e:
            logger.debug("Langfuse get_prompt failed, using registry: %s", e)

    messages = get_default_messages(name)
    # LangChain from_messages accepts list of (role, content) where content may have {vars}
    return ChatPromptTemplate.from_messages(
        [(m["role"], m["content"]) for m in messages]
    )
