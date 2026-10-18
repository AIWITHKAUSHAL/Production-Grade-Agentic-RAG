"""Main components for the agentic RAG application."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import health, ingest, query
from .core.config import get_settings

# ─── Logging Setup ───────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the graph and close checkpoint connections on shutdown."""
    settings = get_settings()
    logger.info("Starting Agentic RAG Assistant", model=settings.euri_model)

    # Pre-warm the LangGraph
    try:
        from .agents.graph import get_graph

        get_graph()
        logger.info("LangGraph compiled successfully")
    except Exception as e:
        logger.error("LangGraph compilation failed", error=str(e))

    try:
        yield
    finally:
        from .checkpoints.mongodb_saver import close_checkpoint_saver

        close_checkpoint_saver()
        logger.info("Shutting down")


# ─── FastAPI App ──────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """Create the API application and register routes and static files."""
    settings = get_settings()

    app = FastAPI(
        title="Production Agentic RAG Assistant",
        description="""
        A production-grade Agentic RAG system with LangGraph, CRAG, Self-RAG,
        Hybrid Search (BM25 + Semantic), RRF, Cross-Encoder Reranking, Qdrant Cloud,
        and MongoDB checkpointing.
        """,
        version="1.0.0",
        lifespan=lifespan,
        debug=settings.debug,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API Routes
    app.include_router(health.router, prefix="/api/v1")
    app.include_router(ingest.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")

    # Serve frontend static files
    frontend_dir = Path(__file__).parent.parent.parent / "frontend"
    if frontend_dir.exists():
        static_dir = frontend_dir / "static"
        if static_dir.exists():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/")
        async def serve_frontend():
            """Serve the chat interface."""
            return FileResponse(str(frontend_dir / "index.html"))

        @app.get("/architecture")
        async def serve_architecture():
            """Serve the architecture page."""
            return FileResponse(str(frontend_dir / "architecture.html"))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
