from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from functools import lru_cache
from app.config import settings
from app.vectorstore.chroma_client import get_chroma_client


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)


def embed_and_store(
    chunks: list[Document],
    collection_name: str | None = None,
) -> int:
    """
    Embed chunks with sentence-transformers and upsert into Chroma.
    Upsert is idempotent — re-uploading the same file is safe.
    Returns the number of chunks stored.
    """
    collection_name = collection_name or settings.CHROMA_COLLECTION
    embedder = get_embedder()

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    embeddings = embedder.embed_documents(texts)

    chroma = get_chroma_client()
    collection = chroma.get_or_create_collection(collection_name)
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return len(chunks)
