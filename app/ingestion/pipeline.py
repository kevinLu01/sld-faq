import asyncio
import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse
from langchain_core.documents import Document
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_and_store


def _assert_safe_url(url: str) -> None:
    """P0: Block SSRF by rejecting private/loopback/link-local targets."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme!r}")
    host = parsed.hostname or ""
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except Exception as exc:
        raise ValueError(f"Cannot resolve host {host!r}: {exc}") from exc
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
        raise ValueError(f"Requests to private/internal addresses are not allowed: {ip}")


async def ingest_file(
    file_path: str,
    original_filename: str,
    mime_type: str,
) -> dict:
    """Load, chunk, and embed a local file. Called as a background task."""
    try:
        docs = await asyncio.to_thread(load_document, file_path, mime_type)
        for doc in docs:
            doc.metadata["original_filename"] = original_filename
        chunks = chunk_documents(docs)
        count = await asyncio.to_thread(embed_and_store, chunks)
        return {"chunks_ingested": count, "filename": original_filename}
    finally:
        # Clean up temp file regardless of success or failure
        Path(file_path).unlink(missing_ok=True)


async def ingest_url(url: str) -> dict:
    """Fetch a URL, treat its text content as a document, and ingest it."""
    import httpx
    from bs4 import BeautifulSoup

    _assert_safe_url(url)  # P0: SSRF guard

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
        response = await client.get(url, headers={"User-Agent": "ResearchAgent/1.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)[:50_000]

    doc = Document(
        page_content=text,
        metadata={"source": url, "type": "web", "original_filename": url},
    )
    chunks = chunk_documents([doc])
    count = await asyncio.to_thread(embed_and_store, chunks)
    return {"chunks_ingested": count, "url": url}
