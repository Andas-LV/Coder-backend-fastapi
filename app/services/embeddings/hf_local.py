import asyncio
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.embeddings.base import EmbeddingProvider

_BATCH_SIZE = 16


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embed_model)


class HFLocalEmbeddings(EmbeddingProvider):
    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = _load_model().encode(
            texts,
            batch_size=_BATCH_SIZE,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
