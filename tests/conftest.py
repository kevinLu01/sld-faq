# ==============================================================================
# Shared pytest fixtures for the kevinai test suite.
#
# How to run:
#   pytest tests/ -v
#   pytest tests/ -v --asyncio-mode=auto   (if not set in pyproject.toml)
#
# External dependencies that are mocked globally here:
#   - app.config.settings  (patched so no real .env is required)
#   - ChromaDB / HuggingFace embedder  (patched per-module as needed)
# ==============================================================================

import os
import sys
import types
from unittest.mock import MagicMock
import pytest

# ---------------------------------------------------------------------------
# Set required env vars BEFORE any app module is imported (including at
# collection time by test_gradio.py's module-level patch() calls).
# ---------------------------------------------------------------------------
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-testing")
os.environ.setdefault("TAVILY_API_KEY", "test-tavily-key")
os.environ.setdefault("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `import app.*` works regardless
# of how pytest is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Stub out heavy / external modules BEFORE any app code is imported so that
# importing app.config or app.agent.* never tries to hit real services.
# ---------------------------------------------------------------------------

def _make_stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# langchain_anthropic
_lca = _make_stub_module("langchain_anthropic")
_lca.ChatAnthropic = object  # replaced per-test with MagicMock when needed

# chromadb
_chroma = _make_stub_module("chromadb")
_chroma.PersistentClient = object

# sentence_transformers / HuggingFace embeddings
_st = _make_stub_module("sentence_transformers")
_lce = _make_stub_module("langchain_community")
_lce_emb = _make_stub_module("langchain_community.embeddings")
_lce_emb.HuggingFaceEmbeddings = object
_lce_vs = _make_stub_module("langchain_community.vectorstores")
_lce_vs.Chroma = object
_lce_doc = _make_stub_module("langchain_community.document_loaders")
_lce_doc.PyPDFLoader = object
_lce_doc.TextLoader = object
sys.modules["langchain_community.embeddings"] = _lce_emb
sys.modules["langchain_community.vectorstores"] = _lce_vs
sys.modules["langchain_community.document_loaders"] = _lce_doc

# tavily
_tav = _make_stub_module("tavily")
_tav.TavilyClient = object

# gradio — stub out so test_gradio.py can patch without installing the full package
_gr = _make_stub_module("gradio")
_gr.Blocks = MagicMock
_gr.Chatbot = MagicMock
_gr.Textbox = MagicMock
_gr.Button = MagicMock
_gr.Slider = MagicMock
_gr.File = MagicMock
_gr.Row = MagicMock
_gr.Tab = MagicMock
_gr.Tabs = MagicMock
_gr.Markdown = MagicMock
_gr.State = MagicMock
_gr_themes = _make_stub_module("gradio.themes")
_gr_themes.Soft = MagicMock
_gr.themes = _gr_themes

# langchain_community.document_loaders sub-modules used by loaders.py
# (already registered above via the parent stub)


# ---------------------------------------------------------------------------
# Patch settings so tests never need a real .env file.
# We expose a fixture that individual tests can reference if they need to
# override specific values.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Replace `app.config.settings` with a simple namespace containing safe
    defaults for every test.  The fixture is *autouse* so no test ever
    accidentally talks to a real external service.
    """
    import types as _types

    fake = _types.SimpleNamespace(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_BASE_URL="https://api.anthropic.com",
        CLAUDE_MODEL="claude-test",
        CLAUDE_MAX_TOKENS=1024,
        CLAUDE_TEMPERATURE=0.0,
        TAVILY_API_KEY="",
        CHROMA_PERSIST_DIR="/tmp/test_chroma",
        CHROMA_COLLECTION="test_collection",
        UPLOAD_DIR="/tmp/test_uploads",
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=50,
        EMBEDDING_MODEL="all-MiniLM-L6-v2",
        VECTOR_SEARCH_K=4,
        MAX_ITERATIONS=3,
        MAX_CONVERSATION_HISTORY_TURNS=3,
        CORS_ORIGINS=["http://localhost:3000"],
        API_HOST="0.0.0.0",
        API_PORT=8000,
    )

    # Patch the module-level singleton used throughout the app
    import app.config as _cfg_mod
    monkeypatch.setattr(_cfg_mod, "settings", fake)

    # Also patch any module that has already imported settings directly
    # (e.g. app.ingestion.chunker, app.memory.conversation)
    for mod_name in list(sys.modules):
        mod = sys.modules[mod_name]
        if hasattr(mod, "settings") and mod_name.startswith("app"):
            try:
                monkeypatch.setattr(mod, "settings", fake)
            except (AttributeError, TypeError):
                pass

    return fake
