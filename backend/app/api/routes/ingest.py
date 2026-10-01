import os
import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from ...core.config import get_settings
from ...ingestion.pipeline import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


class IngestResponse(BaseModel):
    file: str
    chunks: int
    pages: int
    strategy: str
    file_hash: str
    status: str = "success"


@router.post("/upload", response_model=IngestResponse)
async def upload_and_ingest(
    file: UploadFile = File(...),
    chunking_strategy: str = Form("recursive"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    background_tasks: BackgroundTasks = None,
):
    settings = get_settings()
    max_size = settings.max_upload_size_mb * 1024 * 1024

    # Validate file type
    allowed_types = {".pdf", ".txt", ".md", ".docx", ".doc"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_types:
        raise HTTPException(400, f"File type {ext} not supported. Allowed: {allowed_types}")

    # Read and check size
    contents = await file.read()
    if len(contents) > max_size:
        raise HTTPException(413, f"File too large. Max {settings.max_upload_size_mb}MB")

    # Save to disk
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(exist_ok=True)
    file_path = upload_dir / file.filename

    with open(file_path, "wb") as f:
        f.write(contents)

    # Ingest
    try:
        pipeline = get_pipeline()
        result = await pipeline.ingest_file(
            file_path=str(file_path),
            chunking_strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")


@router.get("/strategies")
async def list_chunking_strategies():
    return {
        "strategies": [
            {"id": "recursive", "name": "Recursive (Recommended)", "description": "Splits on hierarchy of separators"},
            {"id": "semantic", "name": "Semantic", "description": "Splits on semantic meaning boundaries"},
            {"id": "sentence", "name": "Sentence", "description": "Splits at sentence boundaries"},
            {"id": "fixed_size", "name": "Fixed Size", "description": "Fixed character count chunks"},
            {"id": "markdown", "name": "Markdown", "description": "Splits on markdown headers"},
            {"id": "paragraph", "name": "Paragraph", "description": "Splits on paragraph boundaries"},
        ]
    }


@router.get("/status")
async def ingest_status():
    from ...retrieval.bm25_retriever import get_bm25_index
    bm25 = get_bm25_index()
    return {"bm25_doc_count": bm25.doc_count, "status": "ready"}
