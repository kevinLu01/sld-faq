# ==============================================================================
# tests/test_api.py
#
# FastAPI route integration tests for:
#   - GET  /health
#   - POST /api/v1/upload
#   - POST /api/v1/ingest-url
#   - POST /api/v1/query
#
# Strategy:
#   • httpx.AsyncClient is used for async endpoint tests.
#   • The LangGraph graph (app.state.graph) is replaced with an AsyncMock.
#   • pipeline.ingest_file / pipeline.ingest_url are patched to avoid I/O.
#   • BackgroundTasks are executed synchronously via TestClient where needed.
# ==============================================================================

import io
import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# App factory – build a fresh FastAPI instance for each test to avoid state
# leaking between tests.  The lifespan (graph build) is replaced by a fixture.
# ---------------------------------------------------------------------------

def _make_app(mock_graph=None):
    """
    Return a FastAPI app with routers mounted and a pre-built mock graph
    attached to app.state so the lifespan is bypassed.
    """
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from app.api.routes import upload, query

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(upload.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")

    @app.get("/health")
    async def health():
        from app.config import settings
        return {"status": "ok", "model": settings.CLAUDE_MODEL}

    # Attach a mock graph (or a default one)
    if mock_graph is None:
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "final_answer": "Test answer",
            "citations": [],
        }
    app.state.graph = mock_graph

    return app


# ---------------------------------------------------------------------------
# Shared async client fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def async_client():
    """Provide an httpx AsyncClient wired to a fresh app instance."""
    app = _make_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client, app


# ===========================================================================
# /health
# ===========================================================================

class TestHealthEndpoint:
    """Tests for the health-check route."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, async_client):
        """GET /health returns HTTP 200 with status='ok'."""
        client, _ = async_client
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_includes_model_name(self, async_client, mock_settings):
        """GET /health response includes the configured model name."""
        client, _ = async_client
        resp = await client.get("/health")
        assert resp.json()["model"] == mock_settings.CLAUDE_MODEL


# ===========================================================================
# POST /api/v1/upload
# ===========================================================================

class TestUploadEndpoint:
    """Tests for the file upload + background ingestion route."""

    def _pdf_file(self, name: str = "test.pdf") -> tuple:
        return (name, io.BytesIO(b"%PDF-1.4 fake content"), "application/pdf")

    @pytest.mark.asyncio
    async def test_upload_pdf_returns_ingesting(self):
        """A valid PDF upload returns 200 with status='ingesting'."""
        app = _make_app()
        with patch("app.api.routes.upload.ingest_file", new_callable=AsyncMock) as mock_ingest, \
             patch("app.api.routes.upload.asyncio.to_thread", new_callable=AsyncMock):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/upload",
                    files={"file": self._pdf_file()},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ingesting"
        assert body["filename"] == "test.pdf"

    @pytest.mark.asyncio
    async def test_upload_text_file_accepted(self):
        """text/plain files are in the supported MIME set."""
        app = _make_app()
        with patch("app.api.routes.upload.ingest_file", new_callable=AsyncMock), \
             patch("app.api.routes.upload.asyncio.to_thread", new_callable=AsyncMock):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/upload",
                    files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
                )
        assert resp.status_code == 200
        assert resp.json()["filename"] == "notes.txt"

    @pytest.mark.asyncio
    async def test_upload_markdown_file_accepted(self):
        """text/markdown files are accepted."""
        app = _make_app()
        with patch("app.api.routes.upload.ingest_file", new_callable=AsyncMock), \
             patch("app.api.routes.upload.asyncio.to_thread", new_callable=AsyncMock):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/upload",
                    files={"file": ("readme.md", io.BytesIO(b"# Title"), "text/markdown")},
                )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_upload_unsupported_mime_returns_415(self):
        """An image/png upload is rejected with HTTP 415 Unsupported Media Type."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/upload",
                files={"file": ("photo.png", io.BytesIO(b"\x89PNG"), "image/png")},
            )
        assert resp.status_code == 415

    @pytest.mark.asyncio
    async def test_upload_json_mime_returns_415(self):
        """application/json is not in the supported set."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/upload",
                files={"file": ("data.json", io.BytesIO(b"{}"), "application/json")},
            )
        assert resp.status_code == 415

    @pytest.mark.asyncio
    async def test_upload_response_contains_message(self):
        """The response body includes an informational message string."""
        app = _make_app()
        with patch("app.api.routes.upload.ingest_file", new_callable=AsyncMock), \
             patch("app.api.routes.upload.asyncio.to_thread", new_callable=AsyncMock):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/upload",
                    files={"file": self._pdf_file()},
                )
        body = resp.json()
        assert body.get("message") is not None and len(body["message"]) > 0

    @pytest.mark.asyncio
    async def test_upload_docx_accepted(self):
        """DOCX MIME type is in the supported set."""
        docx_mime = (
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
        app = _make_app()
        with patch("app.api.routes.upload.ingest_file", new_callable=AsyncMock), \
             patch("app.api.routes.upload.asyncio.to_thread", new_callable=AsyncMock):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/upload",
                    files={"file": ("report.docx", io.BytesIO(b"PK fake"), docx_mime)},
                )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ingesting"


# ===========================================================================
# POST /api/v1/ingest-url
# ===========================================================================

class TestIngestUrlEndpoint:
    """Tests for the URL ingestion route."""

    @pytest.mark.asyncio
    async def test_ingest_url_success(self):
        """A valid URL returns 200 with status='complete' and chunks_ingested."""
        app = _make_app()
        with patch(
            "app.api.routes.upload.ingest_url",
            new_callable=AsyncMock,
            return_value={"chunks_ingested": 7, "url": "https://example.com"},
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/ingest-url",
                    json={"url": "https://example.com"},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "complete"
        assert body["chunks_ingested"] == 7
        assert body["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_ingest_url_error_returns_400(self):
        """If ingest_url raises, the endpoint returns HTTP 400."""
        app = _make_app()
        with patch(
            "app.api.routes.upload.ingest_url",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Connection refused"),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/ingest-url",
                    json={"url": "https://unreachable.example"},
                )

        assert resp.status_code == 400
        assert "Connection refused" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_ingest_url_missing_body_returns_422(self):
        """Sending an empty body returns HTTP 422 Unprocessable Entity."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/v1/ingest-url", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_url_session_id_optional(self):
        """Omitting session_id is valid; a UUID is generated automatically."""
        app = _make_app()
        with patch(
            "app.api.routes.upload.ingest_url",
            new_callable=AsyncMock,
            return_value={"chunks_ingested": 3, "url": "https://docs.example.com"},
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/ingest-url",
                    json={"url": "https://docs.example.com"},
                )
        assert resp.status_code == 200


# ===========================================================================
# POST /api/v1/query
# ===========================================================================

class TestQueryEndpoint:
    """Tests for the synchronous query / RAG endpoint."""

    def _default_graph_return(self, answer="Default answer", citations=None):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "final_answer": answer,
            "citations": citations or [],
        }
        return mock_graph

    @pytest.mark.asyncio
    async def test_query_returns_answer(self):
        """A valid query returns HTTP 200 with the graph's final_answer."""
        mock_graph = self._default_graph_return(answer="Paris is the capital of France.")
        app = _make_app(mock_graph=mock_graph)

        with patch("app.api.routes.query.load_conversation_history", return_value=[]):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/query",
                    json={"query": "What is the capital of France?"},
                )

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_query_response_includes_session_id(self):
        """The response session_id matches the request session_id."""
        mock_graph = self._default_graph_return()
        app = _make_app(mock_graph=mock_graph)

        with patch("app.api.routes.query.load_conversation_history", return_value=[]):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/query",
                    json={"query": "q", "session_id": "explicit-sess"},
                )

        assert resp.json()["session_id"] == "explicit-sess"

    @pytest.mark.asyncio
    async def test_query_citations_in_response(self):
        """Citations from the graph are forwarded to the caller."""
        citation = {
            "source_type": "vector",
            "title": "Doc A",
            "url": None,
            "chunk_id": "doc::chunk_0",
            "score": 0.9,
            "excerpt": "Relevant passage",
        }
        mock_graph = self._default_graph_return(citations=[citation])
        app = _make_app(mock_graph=mock_graph)

        with patch("app.api.routes.query.load_conversation_history", return_value=[]):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/query",
                    json={"query": "What does Doc A say?"},
                )

        body = resp.json()
        assert len(body["citations"]) == 1
        assert body["citations"][0]["title"] == "Doc A"

    @pytest.mark.asyncio
    async def test_query_missing_query_field_returns_422(self):
        """Omitting 'query' in the request body should return HTTP 422."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_query_uses_conversation_history(self):
        """The endpoint loads conversation history and passes it to the graph."""
        mock_graph = self._default_graph_return()
        app = _make_app(mock_graph=mock_graph)
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer", "citations": []},
        ]

        with patch(
            "app.api.routes.query.load_conversation_history",
            return_value=history,
        ) as mock_load:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/query",
                    json={"query": "follow-up question", "session_id": "sess-hist"},
                )

        mock_load.assert_called_once_with("sess-hist")
        # The graph's ainvoke should have been called with the history in state
        call_args = mock_graph.ainvoke.call_args
        initial_state = call_args[0][0]
        assert initial_state["conversation_history"] == history

    @pytest.mark.asyncio
    async def test_query_graph_invoked_with_correct_max_iterations(self):
        """max_iterations from the request is forwarded to the graph state."""
        mock_graph = self._default_graph_return()
        app = _make_app(mock_graph=mock_graph)

        with patch("app.api.routes.query.load_conversation_history", return_value=[]):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post(
                    "/api/v1/query",
                    json={"query": "q", "max_iterations": 2},
                )

        call_args = mock_graph.ainvoke.call_args
        initial_state = call_args[0][0]
        assert initial_state["max_iterations"] == 2

    @pytest.mark.asyncio
    async def test_query_invalid_max_iterations_returns_422(self):
        """max_iterations=0 violates ge=1 constraint — must return 422."""
        app = _make_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/query",
                json={"query": "q", "max_iterations": 0},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_query_graph_no_final_answer_returns_fallback(self):
        """When graph returns no final_answer, the response uses the fallback string."""
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"citations": []}  # no final_answer key
        app = _make_app(mock_graph=mock_graph)

        with patch("app.api.routes.query.load_conversation_history", return_value=[]):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/query",
                    json={"query": "unanswerable question"},
                )

        assert resp.status_code == 200
        assert resp.json()["answer"] == "No answer generated."

    @pytest.mark.asyncio
    async def test_query_initial_state_fields_are_zeroed(self):
        """
        The initial state passed to the graph must have empty collections
        and zeroed counters (iteration_count=0, needs_more_research=False).
        """
        mock_graph = self._default_graph_return()
        app = _make_app(mock_graph=mock_graph)

        with patch("app.api.routes.query.load_conversation_history", return_value=[]):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                await client.post("/api/v1/query", json={"query": "test"})

        state = mock_graph.ainvoke.call_args[0][0]
        assert state["tools_to_use"] == []
        assert state["tools_used"] == []
        assert state["vector_results"] == []
        assert state["web_results"] == []
        assert state["url_results"] == []
        assert state["citations"] == []
        assert state["iteration_count"] == 0
        assert state["needs_more_research"] is False
        assert state["refined_query"] is None
        assert state["final_answer"] is None
