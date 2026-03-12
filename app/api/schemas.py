from typing import Optional
from uuid import uuid4
from pydantic import BaseModel, Field, field_validator
from app.agent.state import SourceCitation


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    max_iterations: int = Field(default=3, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query cannot be blank")
        return v


class QueryResponse(BaseModel):
    answer: str
    citations: list[SourceCitation]
    session_id: str


class UploadResponse(BaseModel):
    status: str           # "ingesting" | "complete" | "error"
    filename: str
    message: Optional[str] = None


class IngestUrlRequest(BaseModel):
    url: str
    session_id: str = Field(default_factory=lambda: str(uuid4()))


class IngestUrlResponse(BaseModel):
    status: str
    url: str
    chunks_ingested: Optional[int] = None
    message: Optional[str] = None
