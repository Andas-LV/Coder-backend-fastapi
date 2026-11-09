from typing import Annotated

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_session
from app.models import User
from app.repositories import chat as repo

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
