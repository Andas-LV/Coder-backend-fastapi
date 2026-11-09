import asyncio
import logging
import threading
from functools import lru_cache

import torch
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.services.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

_BATCH_SIZE = 16
# one encode() at a time: a query embedding must not interleave with a batch of an
# indexing job on the same model/GPU
_encode_lock = threading.Lock()


def resolve_device() -> str:
    """EMBED_DEVICE: 'auto' (CUDA if available, else CPU), 'cuda' or 'cpu'."""
    wanted = settings.embed_device.lower()
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if wanted == "cuda" and not torch.cuda.is_available():
        logger.warning("EMBED_DEVICE=cuda but CUDA is unavailable, falling back to CPU")
        return "cpu"
    return wanted


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    device = resolve_device()
    # half precision on GPU: ~2x faster and half the VRAM, same retrieval quality
    model_kwargs = {"dtype": torch.float16} if device == "cuda" else {}
    model = SentenceTransformer(
        settings.embed_model, device=device, model_kwargs=model_kwargs
    )
    logger.info("Embedding model %s loaded on %s", settings.embed_model, device)
    return model


class HFLocalEmbeddings(EmbeddingProvider):
    def _encode(self, texts: list[str]) -> list[list[float]]:
        with _encode_lock:
            vectors = _load_model().encode(
                texts,
                batch_size=_BATCH_SIZE,
                normalize_embeddings=True,
            )
        return vectors.astype("float32").tolist()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]
