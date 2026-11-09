"""Test doubles shared by the ChatService tests (no network, no database)."""

import uuid

from app.repositories.document import SearchRow
from app.services.llm.base import (
    LLMChunk,
    LLMProvider,
    LLMResult,
    LLMUsage,
    ModelAccess,
)

USER_ID = uuid.uuid4()
DOC_ID = uuid.uuid4()


class FakeProvider(LLMProvider):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, messages, system_prompt, model=None, temperature=None):
        self.calls.append({"messages": messages, "system": system_prompt})
        return LLMResult("Answer [1]", model or "m", LLMUsage(10, 5))

    async def stream(self, messages, system_prompt, model=None):
        self.calls.append({"messages": messages, "system": system_prompt})
        yield LLMChunk(delta="Answer ")
        yield LLMChunk(delta="[1]")
        yield LLMChunk(done=True, usage=LLMUsage(10, 5))

    async def list_models(self):
        return []

    async def check_model(self, model):
        return ModelAccess.FREE


class FakeRetriever:
    def __init__(self, rows: list[SearchRow]) -> None:
        self.rows = rows
        self.calls: list[tuple] = []

    async def __call__(self, user_id, query, document_ids):
        self.calls.append((user_id, query, document_ids))
        return self.rows


def make_row(content="Cats sleep a lot.", page=3, score=0.87654) -> SearchRow:
    return SearchRow(
        content=content,
        page=page,
        position=0,
        document_id=DOC_ID,
        filename="cats.pdf",
        score=score,
    )
