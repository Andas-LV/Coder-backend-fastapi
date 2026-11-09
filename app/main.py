import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import settings
from app.db.session import SessionLocal
from app.repositories import document as document_repo

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # background indexing dies with the process: don't leave documents 'processing' forever
    try:
        async with SessionLocal() as session:
            await document_repo.fail_stale_processing(session)
    except Exception:
        logger.warning(
            "Could not reset stale documents; is the DB migrated? "
            "Run: uv run alembic upgrade head",
            exc_info=True,
        )
    yield


app = FastAPI(title="Coder Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
