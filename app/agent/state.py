import operator
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class SourceCitation(TypedDict):
    source_type: str       # "vector" | "web_search" | "url_fetch"
    title: str
    url: Optional[str]
    chunk_id: Optional[str]
    score: Optional[float]
    excerpt: str


class AgentState(TypedDict):
    # Core conversation — add_messages reducer handles append-only updates
    messages: Annotated[list[BaseMessage], add_messages]

    # Current turn
    query: str
    refined_query: Optional[str]

    # Tool routing
    tools_to_use: list[str]
    # P1: operator.add reducer prevents last-write-wins in parallel tool nodes
    tools_used: Annotated[list[str], operator.add]

    # Retrieved context per source
    vector_results: list[dict]
    web_results: list[dict]
    url_results: list[dict]

    # Output
    citations: list[SourceCitation]
    final_answer: Optional[str]

    # Loop control
    needs_more_research: bool
    iteration_count: int
    max_iterations: int

    # Session
    session_id: str
    conversation_history: list[dict]  # serialized prior turns
