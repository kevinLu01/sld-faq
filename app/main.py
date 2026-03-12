from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import upload, query, stream


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build graph once at startup and share via app.state
    from app.agent.graph import build_graph
    app.state.graph = build_graph()
    yield
    # Cleanup (if needed) goes here


app = FastAPI(
    title="Research Agent API",
    version="1.0.0",
    description="LangGraph-based RAG + Research Agent with local docs, URL crawling, and real-time search.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(stream.router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.CLAUDE_MODEL}
