# ==============================================================================
# tests/test_schemas.py
#
# Unit tests for the Pydantic models in app.api.schemas:
#   - QueryRequest
#   - QueryResponse
#   - UploadResponse
#   - IngestUrlRequest
#   - IngestUrlResponse
#
# Validates both happy-path construction and rejection of invalid inputs.
# ==============================================================================

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Helpers – lazy import so conftest.py stubs are in place first.
# ---------------------------------------------------------------------------

def _import_schemas():
    from app.api import schemas
    return schemas


# ===========================================================================
# QueryRequest
# ===========================================================================

class TestQueryRequest:
    """Validate QueryRequest schema constraints."""

    def test_minimal_valid_request(self):
        """Only 'query' is required; session_id and max_iterations have defaults."""
        s = _import_schemas()
        req = s.QueryRequest(query="Tell me about RAG")
        assert req.query == "Tell me about RAG"
        assert req.max_iterations == 3
        assert isinstance(req.session_id, str) and len(req.session_id) > 0

    def test_session_id_auto_generated_as_uuid(self):
        """Two requests with no explicit session_id get different UUIDs."""
        s = _import_schemas()
        r1 = s.QueryRequest(query="q1")
        r2 = s.QueryRequest(query="q2")
        assert r1.session_id != r2.session_id

    def test_explicit_session_id_is_preserved(self):
        """An explicit session_id is stored as-is."""
        s = _import_schemas()
        req = s.QueryRequest(query="q", session_id="my-session-123")
        assert req.session_id == "my-session-123"

    def test_max_iterations_minimum_boundary(self):
        """max_iterations=1 is the allowed lower bound (ge=1)."""
        s = _import_schemas()
        req = s.QueryRequest(query="q", max_iterations=1)
        assert req.max_iterations == 1

    def test_max_iterations_maximum_boundary(self):
        """max_iterations=5 is the allowed upper bound (le=5)."""
        s = _import_schemas()
        req = s.QueryRequest(query="q", max_iterations=5)
        assert req.max_iterations == 5

    def test_max_iterations_below_min_raises(self):
        """max_iterations=0 violates ge=1 constraint."""
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.QueryRequest(query="q", max_iterations=0)

    def test_max_iterations_above_max_raises(self):
        """max_iterations=6 violates le=5 constraint."""
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.QueryRequest(query="q", max_iterations=6)

    def test_missing_query_raises(self):
        """'query' is a required field; omitting it must raise ValidationError."""
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.QueryRequest()

    def test_empty_query_string_rejected(self):
        """Empty string must be rejected after P0 fix added min_length=1."""
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.QueryRequest(query="")

    def test_negative_max_iterations_raises(self):
        """Negative value is below ge=1 and must be rejected."""
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.QueryRequest(query="q", max_iterations=-1)


# ===========================================================================
# QueryResponse
# ===========================================================================

class TestQueryResponse:
    """Validate QueryResponse schema."""

    def test_valid_response_no_citations(self):
        """A response with an empty citations list is valid."""
        s = _import_schemas()
        resp = s.QueryResponse(
            answer="The answer is 42.",
            citations=[],
            session_id="sess-1",
        )
        assert resp.answer == "The answer is 42."
        assert resp.citations == []

    def test_missing_answer_raises(self):
        """'answer' is required."""
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.QueryResponse(citations=[], session_id="s")

    def test_missing_session_id_raises(self):
        """'session_id' is required."""
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.QueryResponse(answer="a", citations=[])

    def test_citations_with_source_citation_dicts(self):
        """SourceCitation-compatible dicts are accepted inside citations."""
        s = _import_schemas()
        citation = {
            "source_type": "vector",
            "title": "Doc A",
            "url": None,
            "chunk_id": "doc::chunk_0",
            "score": 0.95,
            "excerpt": "Some excerpt text",
        }
        resp = s.QueryResponse(
            answer="answer",
            citations=[citation],
            session_id="s1",
        )
        assert len(resp.citations) == 1


# ===========================================================================
# UploadResponse
# ===========================================================================

class TestUploadResponse:
    """Validate UploadResponse schema."""

    def test_minimal_upload_response(self):
        """status and filename are required; message is optional."""
        s = _import_schemas()
        resp = s.UploadResponse(status="ingesting", filename="report.pdf")
        assert resp.status == "ingesting"
        assert resp.filename == "report.pdf"
        assert resp.message is None

    def test_with_optional_message(self):
        """message field can be set."""
        s = _import_schemas()
        resp = s.UploadResponse(
            status="complete",
            filename="data.txt",
            message="File processed successfully.",
        )
        assert resp.message == "File processed successfully."

    def test_missing_status_raises(self):
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.UploadResponse(filename="f.pdf")

    def test_missing_filename_raises(self):
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.UploadResponse(status="ingesting")

    def test_error_status_accepted(self):
        """status='error' is a valid string value."""
        s = _import_schemas()
        resp = s.UploadResponse(status="error", filename="bad.pdf", message="Failed.")
        assert resp.status == "error"


# ===========================================================================
# IngestUrlRequest
# ===========================================================================

class TestIngestUrlRequest:
    """Validate IngestUrlRequest schema."""

    def test_valid_url_request(self):
        """Only 'url' is required; session_id has a default."""
        s = _import_schemas()
        req = s.IngestUrlRequest(url="https://example.com/docs")
        assert req.url == "https://example.com/docs"
        assert isinstance(req.session_id, str) and len(req.session_id) > 0

    def test_explicit_session_id(self):
        """Explicit session_id is preserved."""
        s = _import_schemas()
        req = s.IngestUrlRequest(url="https://example.com", session_id="fixed-id")
        assert req.session_id == "fixed-id"

    def test_missing_url_raises(self):
        """'url' is required."""
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.IngestUrlRequest()

    def test_unique_session_ids_auto_generated(self):
        """Two requests without explicit session_id get different UUIDs."""
        s = _import_schemas()
        r1 = s.IngestUrlRequest(url="https://a.com")
        r2 = s.IngestUrlRequest(url="https://b.com")
        assert r1.session_id != r2.session_id


# ===========================================================================
# IngestUrlResponse
# ===========================================================================

class TestIngestUrlResponse:
    """Validate IngestUrlResponse schema."""

    def test_minimal_response(self):
        """status and url are required; optional fields default to None."""
        s = _import_schemas()
        resp = s.IngestUrlResponse(status="complete", url="https://example.com")
        assert resp.status == "complete"
        assert resp.chunks_ingested is None
        assert resp.message is None

    def test_full_response(self):
        """All optional fields can be populated."""
        s = _import_schemas()
        resp = s.IngestUrlResponse(
            status="complete",
            url="https://example.com",
            chunks_ingested=12,
            message="OK",
        )
        assert resp.chunks_ingested == 12
        assert resp.message == "OK"

    def test_missing_status_raises(self):
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.IngestUrlResponse(url="https://x.com")

    def test_missing_url_raises(self):
        s = _import_schemas()
        with pytest.raises(ValidationError):
            s.IngestUrlResponse(status="complete")

    def test_error_status_with_message(self):
        """Error status with a message is a valid response."""
        s = _import_schemas()
        resp = s.IngestUrlResponse(
            status="error",
            url="https://broken.com",
            message="Connection refused",
        )
        assert resp.status == "error"
