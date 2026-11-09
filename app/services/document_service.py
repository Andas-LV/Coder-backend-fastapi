import asyncio
import logging
import unicodedata
import uuid

import pymupdf
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import settings
from app.db.session import SessionLocal
from app.models import Document
from app.models.document import STATUS_FAILED, STATUS_READY
from app.repositories import document as repo
from app.services.chunking import split_text
from app.services.embeddings.base import EmbeddingProvider
from app.services.storage import document_file_key, get_file_storage

logger = logging.getLogger(__name__)

EMBED_BATCH = 16  # chunks embedded and committed per step (= progress granularity)

# Strong references to running indexing jobs (asyncio keeps only weak ones).
_tasks: dict[uuid.UUID, asyncio.Task] = {}
# Embedding is CPU-bound: run one document at a time instead of thrashing the CPU.
_embed_lock = asyncio.Lock()


class IngestError(Exception):
    """The uploaded file cannot be turned into searchable chunks."""


def _extract_pages(data: bytes) -> list[str]:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            return [page.get_text() for page in pdf]
    except Exception as e:
        raise IngestError("Invalid or corrupted PDF") from e


def cancel_processing(document_id: uuid.UUID) -> None:
    task = _tasks.get(document_id)
    if task is not None:
        task.cancel()


async def _index(
    document_id: uuid.UUID,
    parts: list[tuple[int, int, str]],
    embeddings: EmbeddingProvider,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Embed chunks batch by batch, committing as we go so progress is visible."""
    async with session_factory() as session:
        try:
            async with _embed_lock:
                for i in range(0, len(parts), EMBED_BATCH):
                    batch = parts[i : i + EMBED_BATCH]
                    vectors = await embeddings.embed_documents([p[2] for p in batch])
                    await repo.add_chunks(
                        session,
                        document_id,
                        [
                            repo.NewChunk(page=pg, position=pos, content=c, embedding=v)
                            for (pg, pos, c), v in zip(batch, vectors, strict=True)
                        ],
                    )
            await repo.set_status(session, document_id, STATUS_READY)
        except asyncio.CancelledError:
            raise  # document deleted: its chunks are gone via ON DELETE CASCADE
        except Exception as e:
            logger.exception("Indexing of document %s failed", document_id)
            await session.rollback()
            await repo.delete_chunks(session, document_id)
            await repo.set_status(session, document_id, STATUS_FAILED, str(e)[:500])
        finally:
            _tasks.pop(document_id, None)


class DocumentService:
    def __init__(
        self,
        session: AsyncSession,
        embeddings: EmbeddingProvider,
        session_factory: async_sessionmaker[AsyncSession] = SessionLocal,
    ) -> None:
        self.session = session
        self.embeddings = embeddings
        self.session_factory = session_factory

    async def ingest(
        self, user_id: uuid.UUID, chat_id: uuid.UUID, filename: str, data: bytes
    ) -> Document:
        """Validate and chunk the PDF now (fast), embed it in the background (slow).

        Raises IngestError for unreadable PDFs / PDFs without text, so the caller can
        answer immediately. Returns the document in status 'processing'.
        """
        pages = await asyncio.to_thread(_extract_pages, data)

        parts: list[tuple[int, int, str]] = []  # (page, position, content)
        position = 0
        for page_no, text in enumerate(pages, start=1):
            for content in split_text(
                text, settings.chunk_size, settings.chunk_overlap
            ):
                parts.append((page_no, position, content))
                position += 1

        if not parts:
            raise IngestError("PDF has no extractable text")

        # Browsers/macOS may send decomposed letters (Й as И + U+0306): store the
        # composed form so the name displays and compares normally.
        filename = unicodedata.normalize("NFC", filename)
        doc = await repo.create_document(
            self.session, user_id, chat_id, filename, len(pages), len(parts)
        )
        try:
            await get_file_storage().save(
                document_file_key(doc.id),
                data,
                filename=filename,
                content_type="application/pdf",
            )
        except Exception:
            await repo.delete_document(self.session, doc.id, user_id)
            raise
        _tasks[doc.id] = asyncio.create_task(
            _index(doc.id, parts, self.embeddings, self.session_factory)
        )
        return doc

    async def search(
        self,
        user_id: uuid.UUID,
        query: str,
        top_k: int,
        document_ids: list[uuid.UUID] | None,
    ) -> list[repo.SearchRow]:
        query_vec = await self.embeddings.embed_query(query)
        return await repo.search_chunks(
            self.session, user_id, query_vec, top_k, document_ids
        )
