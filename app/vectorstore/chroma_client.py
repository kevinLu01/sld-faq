import chromadb
from functools import lru_cache
from app.config import settings


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.PersistentClient:
    """
    Singleton PersistentClient. lru_cache ensures only one client per process,
    avoiding SQLite file lock conflicts with Chroma's backend.
    """
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
