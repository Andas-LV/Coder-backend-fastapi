import uuid
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatCreate(BaseModel):
    title: str = Field(default="New chat", max_length=255)
    system_prompt: str | None = None
    model: str | None = None


class SourceOut(BaseModel):
    n: int  # citation number used in the answer: [n]
    document_id: uuid.UUID
    filename: str
    page: int
    score: float | None  # similarity 0..1; None when the whole document was given


class AttachmentOut(BaseModel):
    document_id: uuid.UUID
    filename: str
    pages: int


class MessageCreate(BaseModel):
    content: str = Field(min_length=1)


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: Literal["user", "assistant", "system"]
    content: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    cost_usd: Decimal | None
    sources: list[SourceOut] | None = None
    attachments: list[AttachmentOut] | None = None
    created_at: datetime


class ChatOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    system_prompt: str | None
    model: str | None
    created_at: datetime
    updated_at: datetime


class ChatDetail(ChatOut):
    messages: list[MessageOut]


class FileLinkOut(BaseModel):
    url: str  # presigned link to the object storage; open directly, no auth header
    filename: str
    expires_in: int  # seconds the link stays valid
