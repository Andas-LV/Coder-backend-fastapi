from fastapi import APIRouter

from app.api.routes import chats, documents, health, models

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chats.router)
api_router.include_router(models.router)
api_router.include_router(documents.router)
