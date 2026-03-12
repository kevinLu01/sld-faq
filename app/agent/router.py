from app.agent.state import AgentState

TOOL_NODE_MAP = {
    "vector_search": "vector_search_node",
    "tavily_search": "tavily_search_node",
    "url_fetch": "url_fetch_node",
}


def route_after_planner(state: AgentState) -> list[str]:
    """Fan-out to all requested tool nodes in parallel."""
    routes = [
        TOOL_NODE_MAP[tool]
        for tool in state.get("tools_to_use", [])
        if tool in TOOL_NODE_MAP
    ]
    # If planner chose no tools, jump straight to synthesizer
    return routes or ["synthesizer"]


def route_after_grader(state: AgentState) -> str:
    """Loop back for more research, or proceed to synthesis."""
    if state.get("needs_more_research") and state.get("iteration_count", 1) < state.get("max_iterations", 3):
        return "planner"
    return "synthesizer"
