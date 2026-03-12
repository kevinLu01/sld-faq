from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_document(file_path: str, mime_type: str) -> list[Document]:
    """Dispatch to the appropriate loader based on MIME type."""
    if mime_type == "application/pdf":
        return PyPDFLoader(file_path).load()

    if mime_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        try:
            import docx2txt
            text = docx2txt.process(file_path)
            return [Document(page_content=text, metadata={"source": file_path})]
        except Exception as e:
            raise ValueError(f"Failed to load DOCX: {e}") from e

    if mime_type in ("text/plain", "text/markdown"):
        return TextLoader(file_path, encoding="utf-8").load()

    raise ValueError(f"Unsupported MIME type: {mime_type}")
