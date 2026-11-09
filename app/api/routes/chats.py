import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import CurrentUser, SessionDep
from app.models import Chat
from app.repositories import chat as repo
from app.schemas.chat import ChatCreate, ChatDetail, ChatOut, MessageCreate, MessageOut
from app.services.chat_service import ChatService
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


@router.post("/{chat_id}/messages", response_model=MessageOut)
async def send_message(
    chat_id: uuid.UUID, body: MessageCreate, session: SessionDep, user: CurrentUser
):
    chat = await _get_chat_or_404(session, chat_id, user)
    try:
        return await ChatService(session, get_llm_provider()).send(chat, body.content)
    except Exception as e:
        raise HTTPException(502, f"LLM provider error: {e}") from e


@router.post("/{chat_id}/messages/stream")
async def stream_message(
    chat_id: uuid.UUID, body: MessageCreate, session: SessionDep, user: CurrentUser
):
    chat = await _get_chat_or_404(session, chat_id, user)
    service = ChatService(session, get_llm_provider())

    async def events():
        try:
            async for event, data in service.stream(chat, body.content):
                payload = "\n".join(f"data: {line}" for line in data.split("\n"))
                yield f"event: {event}\n{payload}\n\n"
        except Exception as e:  # noqa: BLE001
            yield f"event: error\ndata: {e}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
