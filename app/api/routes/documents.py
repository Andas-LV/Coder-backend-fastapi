import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from app.api.deps import CurrentUser, SessionDep
from app.models import Document
from app.repositories import document as repo
from app.schemas.document import (
    DocumentOut,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from app.services.document_service import (
    DocumentService,
    IngestError,
    cancel_processing,
)
from app.services.embeddings import get_embedding_provider

router = APIRouter(tags=["documents"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB


def _out(doc: Document, chunks_count: int) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        pages=doc.pages,
        status=doc.status,
        error=doc.error,
        chunks_total=doc.chunks_total,
        chunks_count=chunks_count,
        created_at=doc.created_at,
    )


@router.post("/documents", response_model=DocumentOut, status_code=202)
async def upload_document(
    file: Annotated[UploadFile, File()], session: SessionDep, user: CurrentUser
):
    if file.content_type != "application/pdf":
        raise HTTPException(415, "Only application/pdf is supported")

    data = await file.read(MAX_UPLOAD_BYTES + 1)  # read one byte over the limit
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 20 MB")

    service = DocumentService(session, get_embedding_provider())
    try:
        doc = await service.ingest(user.id, file.filename or "document.pdf", data)
    except IngestError as e:
        raise HTTPException(422, str(e)) from e
    return _out(doc, 0)


@router.get("/documents/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, session: SessionDep, user: CurrentUser):
    found = await repo.get_document(session, document_id, user.id)
    if found is None:
        raise HTTPException(404, "Document not found")
    return _out(*found)


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents(session: SessionDep, user: CurrentUser):
    return [_out(d, n) for d, n in await repo.list_documents(session, user.id)]


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(
    document_id: uuid.UUID, session: SessionDep, user: CurrentUser
):
    if not await repo.delete_document(session, document_id, user.id):
        raise HTTPException(404, "Document not found")
    cancel_processing(document_id)
    return Response(status_code=204)


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, session: SessionDep, user: CurrentUser):
    service = DocumentService(session, get_embedding_provider())
    rows = await service.search(user.id, body.query, body.top_k, body.document_ids)
    return SearchResponse(
        hits=[
            SearchHit(
                document_id=r.document_id,
                filename=r.filename,
                page=r.page,
                position=r.position,
                content=r.content,
                score=r.score,
            )
            for r in rows
        ]
    )
