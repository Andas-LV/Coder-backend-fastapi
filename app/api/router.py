from fastapi import APIRouter

from app.api.routes import chats, health, models

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chats.router)
api_router.include_router(models.router)
