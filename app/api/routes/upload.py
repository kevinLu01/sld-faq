import asyncio
import os
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, UploadFile, HTTPException

from app.api.schemas import IngestUrlRequest, IngestUrlResponse, UploadResponse
from app.config import settings
from app.ingestion.pipeline import ingest_file, ingest_url

router = APIRouter(tags=["ingestion"])

_SUPPORTED_MIME = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "text/plain",
    "text/markdown",
}


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a PDF, DOCX, or TXT file for ingestion into the knowledge base."""
    mime_type = file.content_type or "application/octet-stream"
    if mime_type not in _SUPPORTED_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {mime_type}. Supported: PDF, DOCX, TXT.",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    save_path = str(upload_dir / f"{uuid4()}_{file.filename}")
    contents = await file.read()

    await asyncio.to_thread(Path(save_path).write_bytes, contents)

    background_tasks.add_task(ingest_file, save_path, file.filename, mime_type)

    return UploadResponse(
        status="ingesting",
        filename=file.filename,
        message="File received. Ingestion is running in the background.",
    )


@router.post("/ingest-url", response_model=IngestUrlResponse)
async def ingest_url_endpoint(request: IngestUrlRequest):
    """Fetch a URL and ingest its content into the knowledge base."""
    try:
        result = await ingest_url(request.url)
        return IngestUrlResponse(
            status="complete",
            url=request.url,
            chunks_ingested=result["chunks_ingested"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
