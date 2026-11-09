from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.services.model_catalog import get_model_catalog

router = APIRouter(prefix="/models", tags=["models"])


class ModelsOut(BaseModel):
    default: str
    models: list[str]


@router.get("", response_model=ModelsOut)
async def list_free_models(refresh: bool = False):
    """Free chat models available right now. Cached (see MODELS_CACHE_TTL_SECONDS);
    pass ?refresh=true to re-probe."""
    try:
        models = await get_model_catalog().free_models(force=refresh)
    except Exception as e:
        raise HTTPException(502, f"LLM provider error: {e}") from e
    return ModelsOut(default=settings.llm_model, models=models)
