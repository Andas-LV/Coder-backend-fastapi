import uuid
from dataclasses import dataclass

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Document
from app.models.document import STATUS_FAILED, STATUS_PROCESSING, STATUS_READY


@dataclass
class NewChunk:
    page: int
    position: int
    content: str
    embedding: list[float]


@dataclass
class SearchRow:
    content: str
    page: int
    position: int
    document_id: uuid.UUID
    filename: str
    score: float


async def create_document(
    session: AsyncSession,
    user_id: uuid.UUID,
    chat_id: uuid.UUID,
    filename: str,
    pages: int,
    chunks_total: int,
) -> Document:
    doc = Document(
        user_id=user_id,
        chat_id=chat_id,
        filename=filename,
        pages=pages,
        chunks_total=chunks_total,
        status=STATUS_PROCESSING,
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)
    return doc


async def add_chunks(
    session: AsyncSession, document_id: uuid.UUID, chunks: list[NewChunk]
) -> None:
    session.add_all(
        Chunk(
            document_id=document_id,
            page=c.page,
            position=c.position,
            content=c.content,
            embedding=c.embedding,
        )
        for c in chunks
    )
    await session.commit()


async def set_status(
    session: AsyncSession,
    document_id: uuid.UUID,
    status: str,
    error: str | None = None,
) -> None:
    await session.execute(
        update(Document)
        .where(Document.id == document_id)
        .values(status=status, error=error)
    )
    await session.commit()


async def delete_chunks(session: AsyncSession, document_id: uuid.UUID) -> None:
    await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
    await session.commit()


async def fail_stale_processing(session: AsyncSession) -> int:
    """Documents left 'processing' by a previous server run will never finish."""
    result = await session.execute(
        update(Document)
        .where(Document.status == STATUS_PROCESSING)
        .values(status=STATUS_FAILED, error="Interrupted by server restart")
    )
    await session.commit()
    return result.rowcount


def _with_counts():
    return (
        select(Document, func.count(Chunk.id))
        .outerjoin(Chunk, Chunk.document_id == Document.id)
        .group_by(Document.id)
    )


async def delete_document(
    session: AsyncSession, document_id: uuid.UUID, user_id: uuid.UUID
) -> bool:
    result = await session.execute(
        delete(Document).where(Document.id == document_id, Document.user_id == user_id)
    )
    await session.commit()
    return result.rowcount > 0


async def search_chunks(
    session: AsyncSession,
    user_id: uuid.UUID,
    query_vec: list[float],
    top_k: int,
    document_ids: list[uuid.UUID] | None = None,
) -> list[SearchRow]:
    distance = Chunk.embedding.cosine_distance(query_vec)
    stmt = (
        select(
            Chunk.content,
            Chunk.page,
            Chunk.position,
            Chunk.document_id,
            Document.filename,
            (1 - distance).label("score"),
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.user_id == user_id)  # only this user's documents
        .where(Document.status == STATUS_READY)  # skip half-indexed documents
        .order_by(distance)
        .limit(top_k)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))
    result = await session.execute(stmt)
    return [
        SearchRow(
            content=r.content,
            page=r.page,
            position=r.position,
            document_id=r.document_id,
            filename=r.filename,
            score=float(r.score),
        )
        for r in result.all()
    ]


async def get_documents_with_counts(
    session: AsyncSession, document_ids: list[uuid.UUID]
) -> list[tuple[Document, int]]:
    """Fresh status + indexed-chunk count for the given documents (progress polling)."""
    result = await session.execute(
        _with_counts()
        .where(Document.id.in_(document_ids))
        # the session may already hold these rows; re-read status/error
        .execution_options(populate_existing=True)
    )
    return [(doc, count) for doc, count in result.all()]
