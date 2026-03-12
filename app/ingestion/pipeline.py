import asyncio
from langchain_core.documents import Document
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_and_store


async def ingest_file(
    file_path: str,
    original_filename: str,
    mime_type: str,
) -> dict:
    """Load, chunk, and embed a local file. Called as a background task."""
    docs = await asyncio.to_thread(load_document, file_path, mime_type)
    for doc in docs:
        doc.metadata["original_filename"] = original_filename
    chunks = chunk_documents(docs)
    count = await asyncio.to_thread(embed_and_store, chunks)
    return {"chunks_ingested": count, "filename": original_filename}


async def ingest_url(url: str) -> dict:
    """Fetch a URL, treat its text content as a document, and ingest it."""
    import httpx
    from bs4 import BeautifulSoup

    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
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
