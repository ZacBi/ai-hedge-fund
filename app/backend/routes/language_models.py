from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any

from app.backend.models.schemas import ErrorResponse
from app.backend.database import get_db
from app.backend.services.ollama_service import OllamaService
from app.backend.services.model_list_service import (
    get_all_models_from_db,
    seed_from_json_if_empty,
    refresh_openrouter_models,
)
from sqlalchemy.orm import Session

router = APIRouter(prefix="/language-models")

ollama_service = OllamaService()


def _cloud_models(db: Session) -> List[Dict[str, Any]]:
    """Cloud models from DB; seed from static JSON if table empty."""
    seed_from_json_if_empty(db)
    return get_all_models_from_db(db)


@router.get(
    path="/",
    responses={
        200: {"description": "List of available language models"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_language_models(db: Session = Depends(get_db)):
    """Get cloud models from DB (with seed if empty) and live Ollama models."""
    try:
        models = _cloud_models(db)
        ollama_models = await ollama_service.get_available_models()
        models.extend(ollama_models)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve models: {str(e)}")


@router.get(
    path="/providers",
    responses={
        200: {"description": "List of available model providers"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def get_language_model_providers(db: Session = Depends(get_db)):
    """Get providers and models from DB (with seed if empty), grouped by provider."""
    try:
        models = _cloud_models(db)
        providers = {}
        for model in models:
            provider_name = model["provider"]
            if provider_name not in providers:
                providers[provider_name] = {"name": provider_name, "models": []}
            providers[provider_name]["models"].append({
                "display_name": model["display_name"],
                "model_name": model["model_name"],
            })
        return {"providers": list(providers.values())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve providers: {str(e)}")


@router.post(
    path="/refresh-openrouter",
    responses={
        200: {"description": "Refreshed OpenRouter models from API"},
        500: {"model": ErrorResponse, "description": "Refresh failed"},
    },
)
async def refresh_openrouter(db: Session = Depends(get_db)):
    """Fetch current model list from OpenRouter and update DB (replaces existing OpenRouter models)."""
    try:
        deleted, inserted = refresh_openrouter_models(db)
        return {"deleted": deleted, "inserted": inserted, "message": "OpenRouter models refreshed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Refresh failed: {str(e)}") 