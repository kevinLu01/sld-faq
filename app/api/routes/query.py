from fastapi import APIRouter, Request
from langchain_core.messages import HumanMessage

from app.api.schemas import QueryRequest, QueryResponse
from app.memory.conversation import load_conversation_history

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest, req: Request):
    """
    Synchronous query endpoint. Runs the full agent graph and returns
    the complete answer with citations.
    """
    graph = req.app.state.graph

    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "query": request.query,
        "session_id": request.session_id,
        "tools_to_use": [],
        "tools_used": [],
        "vector_results": [],
        "web_results": [],
        "url_results": [],
        "citations": [],
        "needs_more_research": False,
        "iteration_count": 0,
        "max_iterations": request.max_iterations,
        "conversation_history": load_conversation_history(request.session_id),
        "refined_query": None,
        "final_answer": None,
    }

    config = {"configurable": {"thread_id": request.session_id}}
    final_state = await graph.ainvoke(initial_state, config=config)

    return QueryResponse(
        answer=final_state.get("final_answer", "No answer generated."),
        citations=final_state.get("citations", []),
        session_id=request.session_id,
    )
