import uuid
from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatCreate(BaseModel):
    title: str = Field(default="New chat", max_length=255)
    system_prompt: str | None = None
    model: str | None = None


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
