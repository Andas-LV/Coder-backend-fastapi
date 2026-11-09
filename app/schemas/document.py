import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    pages: int
    status: Literal["processing", "ready", "failed"]
    error: str | None
    chunks_total: int  # chunks to be indexed
    chunks_count: int  # chunks indexed so far (progress = count / total)
    created_at: datetime


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_ids: list[uuid.UUID] | None = None


class SearchHit(BaseModel):
    document_id: uuid.UUID
    filename: str
    page: int
    position: int
    content: str
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]
