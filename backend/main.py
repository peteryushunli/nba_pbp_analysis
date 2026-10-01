"""FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.db.connection import get_connection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DuckDB views on startup
    get_connection()
    yield


app = FastAPI(
    title="NBA PBP Analysis",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

from backend.routers import health, seasons, efg, ratings, on_off, team_situations

app.include_router(health.router, prefix="/api")
app.include_router(seasons.router, prefix="/api")
app.include_router(efg.router, prefix="/api")
app.include_router(ratings.router, prefix="/api")
app.include_router(on_off.router, prefix="/api")

app.include_router(team_situations.router, prefix="/api")
