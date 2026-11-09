import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.models import Message
from app.services import chat_service
from tests.fakes import USER_ID


@pytest.fixture
def db(monkeypatch):
    """In-memory replacement for the repository functions ChatService uses.

    `doc_chunks` is the full text of the chat's documents (DocChunk list). It is empty by
    default, so documents are treated as "too big to read whole" and searched.
    """
    state = SimpleNamespace(messages=[], document_ids=[], doc_chunks=[])

    async def add_message(session, chat_id, role, content, **kw):
        msg = Message(
            id=uuid.uuid4(),
            chat_id=chat_id,
            role=role,
            content=content,
            created_at=datetime.now(UTC),
            **{
                k: kw.get(k)
                for k in (
                    "model",
                    "input_tokens",
                    "output_tokens",
                    "latency_ms",
                    "cost_usd",
                    "sources",
                    "attachments",
                )
            },
        )
        state.messages.append(msg)
        return msg

    async def get_history(session, chat_id):
        return list(state.messages)

    async def get_chat_document_ids(session, chat_id):
        return list(state.document_ids)

    async def get_documents_text_size(session, document_ids):
        return sum(len(c.content) for c in state.doc_chunks)

    async def get_documents_chunks(session, document_ids):
        return [c for c in state.doc_chunks if c.document_id in document_ids]

    monkeypatch.setattr(
        chat_service.document_repo, "get_documents_text_size", get_documents_text_size
    )
    monkeypatch.setattr(
        chat_service.document_repo, "get_documents_chunks", get_documents_chunks
    )
    monkeypatch.setattr(chat_service.repo, "add_message", add_message)
    monkeypatch.setattr(chat_service.repo, "get_history", get_history)
    monkeypatch.setattr(
        chat_service.repo, "get_chat_document_ids", get_chat_document_ids
    )
    return state


@pytest.fixture
def chat():
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=USER_ID,
        system_prompt="You are a cat expert.",
        model="m",
    )
