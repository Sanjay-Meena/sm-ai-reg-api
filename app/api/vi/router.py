from fastapi import APIRouter
from app.api.v1.routes import health, test
api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(test.router, tags=["Test"])
