from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import settings


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks.
    chunk_size=1000 chars, chunk_overlap=200 chars preserves sentence continuity.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["chunk_id"] = f"{source}::chunk_{i}"
        chunk.metadata["chunk_index"] = i

    return chunks
