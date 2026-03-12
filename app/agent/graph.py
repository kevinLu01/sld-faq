from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from app.agent.state import AgentState
from app.agent.nodes import (
    planner,
    vector_search_node,
    tavily_search_node,
    url_fetch_node,
    grader,
    synthesizer,
    responder,
)
from app.agent.router import route_after_planner, route_after_grader


def build_graph():
    """
    Assemble and compile the LangGraph research agent.

    Graph structure:
      START → planner → [tool nodes in parallel] → grader
        grader → planner (if needs_more_research) or synthesizer
        synthesizer → responder → END
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("planner", planner)
    builder.add_node("vector_search_node", vector_search_node)
    builder.add_node("tavily_search_node", tavily_search_node)
    builder.add_node("url_fetch_node", url_fetch_node)
    builder.add_node("grader", grader)
    builder.add_node("synthesizer", synthesizer)
    builder.add_node("responder", responder)

    # Entry point
    builder.add_edge(START, "planner")

    # Planner fans out to tool nodes (conditional, may be parallel)
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "vector_search_node": "vector_search_node",
            "tavily_search_node": "tavily_search_node",
            "url_fetch_node": "url_fetch_node",
            "synthesizer": "synthesizer",
        },
    )

    # All tool nodes converge to grader
    for tool_node in ("vector_search_node", "tavily_search_node", "url_fetch_node"):
        builder.add_edge(tool_node, "grader")

    # Grader routes back to planner or forward to synthesizer
    builder.add_conditional_edges(
        "grader",
        route_after_grader,
        {"planner": "planner", "synthesizer": "synthesizer"},
    )

    builder.add_edge("synthesizer", "responder")
    builder.add_edge("responder", END)

    # MemorySaver persists state across turns keyed by thread_id (= session_id)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)
