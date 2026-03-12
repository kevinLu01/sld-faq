# pytest tests/test_gradio.py -v

"""
Unit tests for gradio_app.py — all external dependencies are mocked.

Covered functions:
  - _format_citations
  - upload_docs
  - ingest_url_handler
  - new_session
"""

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module-level patch: prevent gradio_app from actually importing heavy deps
# (build_graph, settings, gradio) at collection time.
# ---------------------------------------------------------------------------

_MOCK_SETTINGS = SimpleNamespace(
    CLAUDE_MODEL="claude-test",
    ANTHROPIC_BASE_URL="https://api.anthropic.com",
    EMBEDDING_MODEL="all-MiniLM-L6-v2",
    CHROMA_PERSIST_DIR="./data/chroma_db",
    TAVILY_API_KEY="",
)

_PATCHES = [
    patch("app.config.get_settings", return_value=_MOCK_SETTINGS),
    patch("app.config.settings", _MOCK_SETTINGS),
    patch("app.agent.graph.build_graph", return_value=MagicMock()),
    patch("gradio.Blocks", MagicMock()),
    patch("gradio.themes.Soft", MagicMock()),
    patch("gradio.State", MagicMock()),
    patch("gradio.Tabs", MagicMock()),
    patch("gradio.Tab", MagicMock()),
    patch("gradio.Chatbot", MagicMock()),
    patch("gradio.Row", MagicMock()),
    patch("gradio.Textbox", MagicMock()),
    patch("gradio.Button", MagicMock()),
    patch("gradio.Slider", MagicMock()),
    patch("gradio.File", MagicMock()),
    patch("gradio.Markdown", MagicMock()),
]

for _p in _PATCHES:
    _p.start()

# Now it is safe to import the module under test.
import gradio_app  # noqa: E402  (import after patches)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_run_sync(monkeypatch):
    """
    Patch gradio_app._run so it executes the coroutine with asyncio.run
    instead of the production loop-detection logic.
    """
    monkeypatch.setattr(gradio_app, "_run", lambda coro: asyncio.run(coro))


# ===========================================================================
# _format_citations
# ===========================================================================


class TestFormatCitations:

    def test_format_citations_empty(self):
        """_format_citations([]) must return an empty string."""
        result = gradio_app._format_citations([])
        assert result == ""

    def test_format_citations_vector(self):
        """source_type='vector' must display the 📄 icon and no hyperlink."""
        citations = [{"title": "Local Doc", "source_type": "vector"}]
        result = gradio_app._format_citations(citations)
        assert "📄" in result
        assert "Local Doc" in result
        assert "[1]" in result
        # No URL means no markdown link syntax
        assert "](http" not in result

    def test_format_citations_web_search(self):
        """source_type='web_search' must display 🌐 and a markdown hyperlink."""
        citations = [
            {
                "title": "Some Article",
                "url": "https://example.com/article",
                "source_type": "web_search",
            }
        ]
        result = gradio_app._format_citations(citations)
        assert "🌐" in result
        assert "[Some Article](https://example.com/article)" in result
        assert "[1]" in result

    def test_format_citations_url_fetch(self):
        """source_type='url_fetch' must display the 🔗 icon."""
        citations = [
            {
                "title": "Fetched Page",
                "url": "https://fetched.io/page",
                "source_type": "url_fetch",
            }
        ]
        result = gradio_app._format_citations(citations)
        assert "🔗" in result
        assert "Fetched Page" in result

    def test_format_citations_mixed(self):
        """Mixed source types must receive consecutive [1], [2], [3] indices."""
        citations = [
            {"title": "Vector Doc", "source_type": "vector"},
            {"title": "Web Result", "url": "https://web.io", "source_type": "web_search"},
            {"title": "URL Page", "url": "https://url.io", "source_type": "url_fetch"},
        ]
        result = gradio_app._format_citations(citations)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result
        assert "📄" in result
        assert "🌐" in result
        assert "🔗" in result


# ===========================================================================
# upload_docs
# ===========================================================================


class TestUploadDocs:

    def test_upload_docs_no_files(self):
        """upload_docs(None) must return a non-empty prompt string."""
        result = gradio_app.upload_docs(None)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_upload_docs_success(self, monkeypatch, tmp_path):
        """
        When ingest_file succeeds, upload_docs must return a string that
        contains ✅ and the file's base name.
        """
        _make_run_sync(monkeypatch)

        # Create a real temporary file so Path(f.name).name works correctly.
        tmp_file = tmp_path / "report.pdf"
        tmp_file.write_bytes(b"%PDF fake content")

        fake_file = SimpleNamespace(name=str(tmp_file))

        async def _fake_ingest_file(path, filename, mime):
            return {"chunks_ingested": 7, "filename": filename}

        monkeypatch.setattr(gradio_app, "ingest_file", _fake_ingest_file)

        result = gradio_app.upload_docs([fake_file])
        assert "✅" in result
        assert "report.pdf" in result

    def test_upload_docs_failure(self, monkeypatch, tmp_path):
        """
        When ingest_file raises an exception, upload_docs must return a
        string that contains ❌.
        """
        _make_run_sync(monkeypatch)

        tmp_file = tmp_path / "broken.txt"
        tmp_file.write_text("data", encoding="utf-8")

        fake_file = SimpleNamespace(name=str(tmp_file))

        async def _failing_ingest(path, filename, mime):
            raise RuntimeError("storage unavailable")

        monkeypatch.setattr(gradio_app, "ingest_file", _failing_ingest)

        result = gradio_app.upload_docs([fake_file])
        assert "❌" in result


# ===========================================================================
# ingest_url_handler
# ===========================================================================


class TestIngestUrlHandler:

    def test_ingest_url_empty(self):
        """ingest_url_handler('') must return a non-empty prompt string."""
        result = gradio_app.ingest_url_handler("")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_ingest_url_success(self, monkeypatch):
        """
        When ingest_url returns {chunks_ingested: 5}, the handler must return
        a string containing ✅.
        """
        _make_run_sync(monkeypatch)

        async def _fake_ingest_url(url):
            return {"chunks_ingested": 5, "url": url}

        monkeypatch.setattr(gradio_app, "ingest_url", _fake_ingest_url)

        result = gradio_app.ingest_url_handler("https://example.com/doc")
        assert "✅" in result
        assert "5" in result

    def test_ingest_url_failure(self, monkeypatch):
        """
        When ingest_url raises an exception, the handler must return a string
        containing ❌.
        """
        _make_run_sync(monkeypatch)

        async def _failing_url(url):
            raise ConnectionError("network error")

        monkeypatch.setattr(gradio_app, "ingest_url", _failing_url)

        result = gradio_app.ingest_url_handler("https://unreachable.example")
        assert "❌" in result


# ===========================================================================
# new_session
# ===========================================================================


class TestNewSession:

    def test_new_session_returns_uuid(self):
        """
        new_session() must return a valid UUID v4 string, and two consecutive
        calls must produce different values.
        """
        s1 = gradio_app.new_session()
        s2 = gradio_app.new_session()

        # Must be parseable as a UUID
        parsed1 = uuid.UUID(s1, version=4)
        parsed2 = uuid.UUID(s2, version=4)

        # Version must be 4
        assert parsed1.version == 4
        assert parsed2.version == 4

        # Must be unique across calls
        assert s1 != s2
