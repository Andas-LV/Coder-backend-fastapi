import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, MessagePayloadDep, SessionDep
from app.models import Chat
from app.repositories import chat as repo
from app.schemas.chat import (
    ChatCreate,
    ChatDetail,
    ChatOut,
    MessageOut,
)
from app.services.chat_service import ChatService, IndexingFailed
from app.services.document_service import IngestError
from app.services.llm import get_llm_provider

router = APIRouter(prefix="/chats", tags=["chats"])


async def _get_chat_or_404(
    session: SessionDep,
    chat_id: uuid.UUID,
    user: CurrentUser,
    with_messages: bool = False,
) -> Chat:
    chat = await repo.get_chat(session, chat_id, user.id, with_messages=with_messages)
    if chat is None:
        raise HTTPException(404, "Chat not found")
    return chat


@router.post("", response_model=ChatOut, status_code=201)
async def create_chat(body: ChatCreate, session: SessionDep, user: CurrentUser):
    return await repo.create_chat(
        session, user.id, body.title, body.system_prompt, body.model
    )


@router.get("", response_model=list[ChatOut])
async def list_chats(session: SessionDep, user: CurrentUser):
    return await repo.list_chats(session, user.id)


@router.get("/{chat_id}", response_model=ChatDetail)
async def get_chat(chat_id: uuid.UUID, session: SessionDep, user: CurrentUser):
    return await _get_chat_or_404(session, chat_id, user, with_messages=True)


# The message endpoints take JSON {"content"} or multipart (`content` + `files`).
MESSAGE_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["content"],
                    "properties": {"content": {"type": "string", "minLength": 1}},
                }
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["content"],
                    "properties": {
                        "content": {"type": "string", "minLength": 1},
                        "files": {
                            "type": "array",
                            "items": {"type": "string", "format": "binary"},
                            "description": "Up to 5 PDFs, 20 MB each",
                        },
                    },
                }
            },
        },
    }
}


async def _attach(
    service: ChatService, chat_id: uuid.UUID, user: CurrentUser, files
) -> list[dict] | None:
    if not files:
        return None
    try:
        return await service.attach_files(chat_id, user.id, files)
    except IngestError as e:
        raise HTTPException(422, str(e)) from e


@router.post(
    "/{chat_id}/messages", response_model=MessageOut, openapi_extra=MESSAGE_BODY
)
async def send_message(
    chat_id: uuid.UUID,
    payload: MessagePayloadDep,
    session: SessionDep,
    user: CurrentUser,
):
    """Send a message, optionally with PDFs. Waits until attached files are indexed."""
    chat = await _get_chat_or_404(session, chat_id, user)
    service = ChatService(session, get_llm_provider())
    attachments = await _attach(service, chat_id, user, payload.files)
    try:
        return await service.send(chat, payload.content, attachments)
    except IndexingFailed as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        raise HTTPException(502, f"LLM provider error: {e}") from e


@router.post("/{chat_id}/messages/stream", openapi_extra=MESSAGE_BODY)
async def stream_message(
    chat_id: uuid.UUID,
    payload: MessagePayloadDep,
    session: SessionDep,
    user: CurrentUser,
):
    """Send a message (optionally with PDFs) and stream the answer over SSE.

    Events: `indexing` (while attached files are processed), `sources`, `delta`,
    `done`, `error`.
    """
    chat = await _get_chat_or_404(session, chat_id, user)
    service = ChatService(session, get_llm_provider())
    attachments = await _attach(service, chat_id, user, payload.files)

    async def events():
        try:
            async for event, data in service.stream(chat, payload.content, attachments):
                lines = "\n".join(f"data: {line}" for line in data.split("\n"))
                yield f"event: {event}\n{lines}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
