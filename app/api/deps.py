from dataclasses import dataclass, field
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import UploadFile as FormUpload

from app.api.uploads import read_pdf
from app.core.config import settings
from app.db.session import get_session
from app.models import User
from app.repositories import chat as repo
from app.schemas.chat import MessageCreate

SessionDep = Annotated[AsyncSession, Depends(get_session)]

# auto_error=False: return our own 401 instead of FastAPI's default 403
bearer_scheme = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def decode_nextauth_token(token: str) -> dict:
    """Verify the NextAuth session JWT.

    The frontend overrides NextAuth's jwt.encode (core/lib/next-auth.ts) to sign
    an HS256 JWT with the raw NEXTAUTH_SECRET string as the key, so we verify
    it the same way. Payload carries id, email, name, image, provider.
    """
    try:
        return jwt.decode(
            token,
            settings.nextauth_secret.encode(),
            algorithms=["HS256"],
            options={"require": ["exp"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise _unauthorized("Token expired") from e
    except jwt.InvalidTokenError as e:
        raise _unauthorized("Invalid token") from e


async def get_current_user(
    session: SessionDep,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> User:
    if credentials is None:
        raise _unauthorized("Missing bearer token")

    payload = decode_nextauth_token(credentials.credentials)
    email = payload.get("email")
    if not isinstance(email, str) or not email:
        raise _unauthorized("Token has no email")

    user = await repo.get_or_create_user(
        session, email, name=payload.get("name"), image=payload.get("image")
    )
    await session.commit()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


MAX_FILES_PER_MESSAGE = 5


@dataclass
class MessagePayload:
    content: str
    files: list[tuple[str, bytes]] = field(
        default_factory=list
    )  # (filename, pdf bytes)


async def get_message_payload(request: Request) -> MessagePayload:
    """A chat message: JSON {"content"} or multipart (`content` + optional `files`).

    Files are PDFs attached to the message; they are validated here (415 / 413).
    """
    if request.headers.get("content-type", "").startswith("multipart/form-data"):
        form = await request.form()
        raw_content = form.get("content")
        uploads = [f for f in form.getlist("files") if isinstance(f, FormUpload)]
        if len(uploads) > MAX_FILES_PER_MESSAGE:
            raise HTTPException(
                400, f"At most {MAX_FILES_PER_MESSAGE} files per message"
            )
        files = [(u.filename or "document.pdf", await read_pdf(u)) for u in uploads]
    else:
        try:
            raw_content = (await request.json()).get("content")
        except (ValueError, AttributeError) as e:
            raise HTTPException(422, "Body must be JSON with a 'content' field") from e
        files = []
    try:
        body = MessageCreate(
            content=raw_content if isinstance(raw_content, str) else ""
        )
    except ValidationError as e:
        raise HTTPException(422, "'content' must be a non-empty string") from e
    return MessagePayload(body.content, files)


MessagePayloadDep = Annotated[MessagePayload, Depends(get_message_payload)]
