import asyncio
import time

from app.core.config import settings
from app.services.llm import get_llm_provider
from app.services.llm.base import LLMProvider, ModelAccess

_CONCURRENCY = 3  # Ollama returns 429 on too many parallel requests
_RETRIES = 3


class ModelCatalog:
    """Free chat models of the provider, discovered by probing and cached in memory."""

    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider
        self._free: list[str] = []
        self._checked_at: float | None = None
        self._lock = asyncio.Lock()

    def _fresh(self) -> bool:
        return (
            self._checked_at is not None
            and time.monotonic() - self._checked_at < settings.models_cache_ttl_seconds
        )

    async def _check(self, sem: asyncio.Semaphore, model: str) -> ModelAccess:
        async with sem:
            for attempt in range(_RETRIES):
                access = await self._provider.check_model(model)
                if access is not ModelAccess.UNKNOWN:
                    return access
                await asyncio.sleep(1 + attempt)
        return ModelAccess.UNKNOWN

    async def _refresh(self) -> None:
        names = await self._provider.list_models()
        sem = asyncio.Semaphore(_CONCURRENCY)
        results = await asyncio.gather(*(self._check(sem, n) for n in names))
        free = [
            n
            for n, access in zip(names, results, strict=True)
            if access is ModelAccess.FREE
        ]
        # If every probe failed transiently, keep the previous list instead of emptying it.
        if free or not self._free:
            self._free = free
        self._checked_at = time.monotonic()

    async def free_models(self, force: bool = False) -> list[str]:
        if force or not self._fresh():
            async with self._lock:
                if force or not self._fresh():
                    await self._refresh()
        return self._free


_catalog: ModelCatalog | None = None


def get_model_catalog() -> ModelCatalog:
    global _catalog
    if _catalog is None:
        _catalog = ModelCatalog(get_llm_provider())
    return _catalog
