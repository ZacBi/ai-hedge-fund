"""
Model list service: read/write LLM models from DB, seed from static JSON, refresh from OpenRouter.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.backend.database.models import LLMModel as LLMModelRow

logger = logging.getLogger(__name__)

# Path to static api_models.json (used for seed when DB is empty)
API_MODELS_JSON_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "llm" / "api_models.json"
)
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def get_all_models_from_db(db: Session) -> list[dict[str, Any]]:
    """Return all enabled models from DB as list of {display_name, model_name, provider}."""
    rows = (
        db.query(LLMModelRow)
        .filter(LLMModelRow.is_enabled.is_(True))
        .order_by(LLMModelRow.sort_order, LLMModelRow.id)
        .all()
    )
    return [
        {
            "display_name": r.display_name,
            "model_name": r.model_name,
            "provider": r.provider,
        }
        for r in rows
    ]


def seed_from_json_if_empty(db: Session) -> int:
    """
    If llm_models table is empty, seed from api_models.json. Returns number of rows inserted.
    """
    count = db.query(LLMModelRow).count()
    if count > 0:
        return 0
    if not API_MODELS_JSON_PATH.exists():
        logger.warning("api_models.json not found at %s, skipping seed", API_MODELS_JSON_PATH)
        return 0
    with open(API_MODELS_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        display_name = item.get("display_name") or ""
        model_name = item.get("model_name") or ""
        provider = item.get("provider") or "OpenRouter"
        if not display_name or not model_name:
            continue
        row = LLMModelRow(
            display_name=display_name,
            model_name=model_name,
            provider=provider,
            sort_order=i,
            is_enabled=True,
            source="static",
        )
        db.add(row)
    db.commit()
    logger.info("Seeded llm_models from api_models.json: %s rows", len(data))
    return len(data)


def refresh_openrouter_models(db: Session) -> tuple[int, int]:
    """
    Fetch current models from OpenRouter API and upsert into DB.
    Replaces all existing rows with provider=OpenRouter, then inserts new from API.
    Returns (deleted_count, inserted_count).
    """
    try:
        import urllib.request

        req = urllib.request.Request(OPENROUTER_MODELS_URL)
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
    except Exception as e:
        logger.exception("Failed to fetch OpenRouter models: %s", e)
        raise

    data = body.get("data") or []
    # OpenRouter returns list of { id, name, ... }; we use id as model_name, name as display_name
    to_insert = []
    for m in data:
        model_id = m.get("id")
        name = m.get("name") or model_id
        if not model_id:
            continue
        to_insert.append({"model_name": model_id, "display_name": name, "provider": "OpenRouter"})

    deleted = db.query(LLMModelRow).filter(LLMModelRow.provider == "OpenRouter").delete()
    db.commit()

    for i, item in enumerate(to_insert):
        row = LLMModelRow(
            display_name=item["display_name"],
            model_name=item["model_name"],
            provider="OpenRouter",
            sort_order=i,
            is_enabled=True,
            source="openrouter",
        )
        db.add(row)
    db.commit()
    logger.info("Refresh OpenRouter: deleted=%s, inserted=%s", deleted, len(to_insert))
    return deleted, len(to_insert)
