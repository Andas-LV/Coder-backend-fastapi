from app.services.embeddings.base import EmbeddingProvider
from app.services.embeddings.hf_local import HFLocalEmbeddings

_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        _provider = HFLocalEmbeddings()
    return _provider
