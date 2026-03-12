from langchain_core.tools import tool
from app.config import settings
from app.vectorstore.chroma_client import get_chroma_client
from app.ingestion.embedder import get_embedder


@tool
async def vector_search(query: str, k: int = 6) -> list[dict]:
    """
    Search the local Chroma vector store for documents relevant to the query.
    Returns up to k chunks with content, metadata, and relevance score.
    Use this when the user's question may be answered by uploaded documents.
    """
    import asyncio

    embedder = get_embedder()
    query_embedding = await asyncio.to_thread(embedder.embed_query, query)

    chroma = get_chroma_client()
    collection = chroma.get_or_create_collection(settings.CHROMA_COLLECTION)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )

    items = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        items.append({
            "content": doc,
            "metadata": meta,
            "score": round(1 - dist, 4),  # cosine: higher = more relevant
        })

    return items
