#!/usr/bin/env -S uv run python
"""
Sync local default prompts from src.prompts.registry to Langfuse.

Only pushes prompts that have changed (compared to the current production version).
Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in the environment (or .env).
Prompts are created/updated with label "production".

Usage:
  uv run scripts/sync_prompts_to_langfuse.py
  # or from project root:
  python -m scripts.sync_prompts_to_langfuse
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LABEL = "production"


def _langfuse_content(content: str) -> str:
    """Convert LangChain placeholders {var} to Langfuse placeholders {{var}}."""
    for var in (
        "ticker",
        "analysis_data",
        "facts",
        "confidence",
        "signals",
        "allowed",
        "context",
        "company_context_block",
    ):
        content = content.replace("{" + var + "}", "{{" + var + "}}")
    return content


def _local_prompt_messages(messages: list[dict]) -> list[dict]:
    """Build list of {role, content} for Langfuse from registry messages."""
    return [
        {"role": m["role"], "content": _langfuse_content(m["content"])}
        for m in messages
    ]


def _remote_prompt_messages(pf) -> list[dict] | None:
    """Extract list of {role, content} from Langfuse prompt client. None if not chat."""
    if not hasattr(pf, "prompt"):
        return None
    out = []
    for m in pf.prompt:
        t = m.get("type") if isinstance(m, dict) else getattr(m, "type", None)
        if t != "message":
            continue
        role = m.get("role", "") if isinstance(m, dict) else getattr(m, "role", "")
        content = (
            m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
        )
        out.append({"role": role, "content": content})
    return out if out else None


def _prompt_messages_equal(a: list[dict], b: list[dict]) -> bool:
    if len(a) != len(b):
        return False
    for i, (ma, mb) in enumerate(zip(a, b)):
        if ma.get("role") != mb.get("role") or ma.get("content") != mb.get("content"):
            return False
    return True


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
        print("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set. Skipping sync.")
        return 0

    from langfuse import get_client
    from src.prompts.registry import DEFAULT_PROMPTS

    client = get_client()
    updated = 0
    skipped = 0
    for name, messages in DEFAULT_PROMPTS.items():
        local = _local_prompt_messages(messages)
        try:
            pf = client.get_prompt(name, label=LABEL)
            remote = _remote_prompt_messages(pf)
            if remote is not None and _prompt_messages_equal(local, remote):
                print(f"Skipped (unchanged): {name}")
                skipped += 1
                continue
        except Exception:
            # Prompt missing or not chat → will create/update
            pass
        try:
            client.create_prompt(
                name=name,
                type="chat",
                prompt=local,  # type: ignore[arg-type]
                labels=[LABEL],
            )
            print(f"Created/updated prompt: {name}")
            updated += 1
        except Exception as e:
            print(f"Failed {name}: {e}", file=sys.stderr)
            return 1
    print(f"Done. Updated {updated}, skipped {skipped} unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
