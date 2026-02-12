"""
Centralized prompt definitions and loader.
Defaults live in the registry; at runtime prompts are loaded from Langfuse when configured,
otherwise from the local registry.
"""
from src.prompts.loader import get_prompt_template
from src.prompts.registry import PROMPT_NAMES

__all__ = ["get_prompt_template", "PROMPT_NAMES"]
