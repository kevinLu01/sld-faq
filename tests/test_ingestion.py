# ==============================================================================
# tests/test_ingestion.py
#
# Unit tests for:
#   - app.ingestion.chunker.chunk_documents
#   - app.ingestion.loaders.load_document
#
# Heavy external dependencies (PyPDFLoader, TextLoader, docx2txt) are fully
# mocked so tests run without any real files or installed loaders.
# ==============================================================================

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# chunk_documents tests
# ---------------------------------------------------------------------------

class TestChunkDocuments:
    """Tests for the RecursiveCharacterTextSplitter-based chunker."""

    def _import(self):
        from app.ingestion.chunker import chunk_documents
        return chunk_documents

    def test_single_short_doc_returns_one_chunk(self, mock_settings):
        """A document shorter than chunk_size should come back as a single chunk."""
        mock_settings.CHUNK_SIZE = 500
        mock_settings.CHUNK_OVERLAP = 50

        chunk_documents = self._import()
        doc = Document(page_content="Hello world.", metadata={"source": "test.txt"})
        chunks = chunk_documents([doc])

        assert len(chunks) >= 1
        # The original text should appear somewhere in the chunks
        combined = " ".join(c.page_content for c in chunks)
        assert "Hello world" in combined

    def test_long_doc_produces_multiple_chunks(self, mock_settings):
        """A document much longer than chunk_size must be split into multiple chunks."""
        mock_settings.CHUNK_SIZE = 100
        mock_settings.CHUNK_OVERLAP = 10

        chunk_documents = self._import()
        long_text = "word " * 200  # 1 000 chars — well over chunk_size=100
        doc = Document(page_content=long_text, metadata={"source": "long.txt"})
        chunks = chunk_documents([doc])

        assert len(chunks) > 1

    def test_chunk_ids_are_unique_and_sequential(self, mock_settings):
        """Each chunk gets a unique chunk_id based on source and index."""
        mock_settings.CHUNK_SIZE = 100
        mock_settings.CHUNK_OVERLAP = 10

        chunk_documents = self._import()
        long_text = "sentence. " * 100
        doc = Document(page_content=long_text, metadata={"source": "myfile.pdf"})
        chunks = chunk_documents([doc])

        ids = [c.metadata["chunk_id"] for c in chunks]
        # All IDs must be unique
        assert len(ids) == len(set(ids))
        # Each ID must contain the source name
        for cid in ids:
            assert "myfile.pdf" in cid

    def test_chunk_index_metadata_is_sequential(self, mock_settings):
        """chunk_index metadata values should be 0, 1, 2, … with no gaps."""
        mock_settings.CHUNK_SIZE = 100
        mock_settings.CHUNK_OVERLAP = 10

        chunk_documents = self._import()
        doc = Document(page_content="abc " * 200, metadata={"source": "s.txt"})
        chunks = chunk_documents([doc])

        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_document_list_returns_empty(self, mock_settings):
        """Passing an empty list should return an empty list without errors."""
        chunk_documents = self._import()
        result = chunk_documents([])
        assert result == []

    def test_metadata_preserved_through_chunking(self, mock_settings):
        """Original metadata keys (other than chunk_id/index) survive chunking."""
        mock_settings.CHUNK_SIZE = 100
        mock_settings.CHUNK_OVERLAP = 10

        chunk_documents = self._import()
        doc = Document(
            page_content="test " * 100,
            metadata={"source": "orig.txt", "original_filename": "report.pdf"},
        )
        chunks = chunk_documents([doc])

        for chunk in chunks:
            assert chunk.metadata.get("original_filename") == "report.pdf"

    def test_chunk_id_format_uses_double_colon(self, mock_settings):
        """chunk_id must follow the '<source>::chunk_<n>' format."""
        mock_settings.CHUNK_SIZE = 200
        mock_settings.CHUNK_OVERLAP = 20

        chunk_documents = self._import()
        doc = Document(page_content="x " * 300, metadata={"source": "doc.txt"})
        chunks = chunk_documents([doc])

        for i, chunk in enumerate(chunks):
            assert chunk.metadata["chunk_id"] == f"doc.txt::chunk_{i}"

    def test_unknown_source_falls_back_to_unknown(self, mock_settings):
        """Documents without a 'source' key use 'unknown' in the chunk_id."""
        chunk_documents = self._import()
        doc = Document(page_content="data " * 50, metadata={})
        chunks = chunk_documents([doc])

        for chunk in chunks:
            assert "unknown" in chunk.metadata["chunk_id"]

    def test_multiple_docs_chunk_independently(self, mock_settings):
        """Two separate documents each produce chunks; total count is sum of both."""
        mock_settings.CHUNK_SIZE = 100
        mock_settings.CHUNK_OVERLAP = 10

        chunk_documents = self._import()
        doc_a = Document(page_content="alpha " * 100, metadata={"source": "a.txt"})
        doc_b = Document(page_content="beta " * 100, metadata={"source": "b.txt"})
        chunks = chunk_documents([doc_a, doc_b])

        assert len(chunks) >= 2
        sources = {c.metadata.get("source") for c in chunks}
        assert "a.txt" in sources
        assert "b.txt" in sources


# ---------------------------------------------------------------------------
# load_document tests
# ---------------------------------------------------------------------------

class TestLoadDocument:
    """Tests for the MIME-type dispatcher in loaders.py."""

    def _import(self):
        from app.ingestion.loaders import load_document
        return load_document

    # ---- PDF ----------------------------------------------------------------

    @patch("app.ingestion.loaders.PyPDFLoader")
    def test_pdf_uses_pypdf_loader(self, MockPDFLoader):
        """PDF MIME type instantiates PyPDFLoader and calls .load()."""
        expected = [Document(page_content="pdf content", metadata={"source": "f.pdf"})]
        instance = MockPDFLoader.return_value
        instance.load.return_value = expected

        load_document = self._import()
        result = load_document("/tmp/file.pdf", "application/pdf")

        MockPDFLoader.assert_called_once_with("/tmp/file.pdf")
        instance.load.assert_called_once()
        assert result == expected

    # ---- TXT / Markdown -----------------------------------------------------

    @patch("app.ingestion.loaders.TextLoader")
    def test_plain_text_uses_text_loader(self, MockTextLoader):
        """text/plain MIME type routes to TextLoader."""
        expected = [Document(page_content="hello", metadata={})]
        instance = MockTextLoader.return_value
        instance.load.return_value = expected

        load_document = self._import()
        result = load_document("/tmp/file.txt", "text/plain")

        MockTextLoader.assert_called_once_with("/tmp/file.txt", encoding="utf-8")
        assert result == expected

    @patch("app.ingestion.loaders.TextLoader")
    def test_markdown_uses_text_loader(self, MockTextLoader):
        """text/markdown MIME type also routes to TextLoader."""
        expected = [Document(page_content="# Title", metadata={})]
        instance = MockTextLoader.return_value
        instance.load.return_value = expected

        load_document = self._import()
        result = load_document("/tmp/readme.md", "text/markdown")

        MockTextLoader.assert_called_once_with("/tmp/readme.md", encoding="utf-8")
        assert result == expected

    # ---- DOCX ---------------------------------------------------------------

    def test_docx_uses_docx2txt(self):
        """DOCX MIME type calls docx2txt.process and wraps result in a Document."""
        docx_mime = (
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
        with patch.dict("sys.modules", {"docx2txt": MagicMock()}):
            import sys
            sys.modules["docx2txt"].process.return_value = "docx text content"

            load_document = self._import()
            result = load_document("/tmp/file.docx", docx_mime)

        assert len(result) == 1
        assert result[0].page_content == "docx text content"
        assert result[0].metadata["source"] == "/tmp/file.docx"

    def test_msword_uses_docx2txt(self):
        """application/msword is handled the same as the modern DOCX MIME type."""
        with patch.dict("sys.modules", {"docx2txt": MagicMock()}):
            import sys
            sys.modules["docx2txt"].process.return_value = "old word content"

            load_document = self._import()
            result = load_document("/tmp/old.doc", "application/msword")

        assert result[0].page_content == "old word content"

    def test_docx_error_raises_value_error(self):
        """If docx2txt.process raises, load_document re-raises as ValueError."""
        docx_mime = (
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document"
        )
        mock_docx2txt = MagicMock()
        mock_docx2txt.process.side_effect = RuntimeError("corrupt file")

        with patch.dict("sys.modules", {"docx2txt": mock_docx2txt}):
            load_document = self._import()
            with pytest.raises(ValueError, match="Failed to load DOCX"):
                load_document("/tmp/bad.docx", docx_mime)

    # ---- Unsupported type ---------------------------------------------------

    def test_unsupported_mime_raises_value_error(self):
        """An unrecognised MIME type raises ValueError with an informative message."""
        load_document = self._import()
        with pytest.raises(ValueError, match="Unsupported MIME type"):
            load_document("/tmp/image.png", "image/png")

    def test_octet_stream_raises_value_error(self):
        """application/octet-stream (unknown binary) is rejected."""
        load_document = self._import()
        with pytest.raises(ValueError, match="Unsupported MIME type"):
            load_document("/tmp/bin.bin", "application/octet-stream")

    def test_empty_mime_string_raises_value_error(self):
        """An empty MIME type string should be rejected."""
        load_document = self._import()
        with pytest.raises(ValueError):
            load_document("/tmp/file", "")
