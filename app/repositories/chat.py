import uuid
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models import Chat, Message, User


async def get_or_create_user(
    session: AsyncSession,
    email: str,
    name: str | None = None,
    image: str | None = None,
) -> User:
    user = await session.scalar(select(User).where(User.email == email))
    if user is None:
        user = User(email=email, name=name, image=image)
        session.add(user)
        await session.flush()
    else:
        # keep profile in sync with the OAuth provider (replaces POST /users/sync)
        if name and user.name != name:
            user.name = name
        if image and user.image != image:
            user.image = image
    return user


async def create_chat(
    session: AsyncSession,
    user_id: uuid.UUID,
    title: str,
    system_prompt: str | None,
    model: str | None,
) -> Chat:
    chat = Chat(user_id=user_id, title=title, system_prompt=system_prompt, model=model)
    session.add(chat)
    await session.commit()
    await session.refresh(chat)
    return chat


async def list_chats(session: AsyncSession, user_id: uuid.UUID) -> list[Chat]:
    result = await session.scalars(
        select(Chat).where(Chat.user_id == user_id).order_by(Chat.updated_at.desc())
    )
    return list(result)


async def get_chat(
    session: AsyncSession,
    chat_id: uuid.UUID,
    user_id: uuid.UUID,
    with_messages: bool = False,
) -> Chat | None:
    stmt = select(Chat).where(Chat.id == chat_id, Chat.user_id == user_id)
    if with_messages:
        stmt = stmt.options(selectinload(Chat.messages))
    return await session.scalar(stmt)


async def get_history(session: AsyncSession, chat_id: uuid.UUID) -> list[Message]:
    result = await session.scalars(
        select(Message)
        .where(Message.chat_id == chat_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(result)


async def add_message(
    session: AsyncSession,
    chat_id: uuid.UUID,
    role: str,
    content: str,
    *,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: int | None = None,
    cost_usd: Decimal | None = None,
) -> Message:
    msg = Message(
        chat_id=chat_id,
        role=role,
        content=content,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=cost_usd,
    )
    session.add(msg)
    await session.commit()
    await session.refresh(msg)
    return msg
