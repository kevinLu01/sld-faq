# ==============================================================================
# tests/test_router.py
#
# Unit tests for app.agent.router:
#   - route_after_planner  (fan-out to tool nodes)
#   - route_after_grader   (loop-back or proceed to synthesizer)
#
# These functions are pure state-transforming logic with no I/O, so no mocking
# of external services is required beyond the autouse mock_settings fixture.
# ==============================================================================

import pytest
from app.agent.router import route_after_planner, route_after_grader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides) -> dict:
    """Return a minimal AgentState-compatible dict."""
    state = {
        "query": "What is LangGraph?",
        "refined_query": None,
        "tools_to_use": [],
        "tools_used": [],
        "vector_results": [],
        "web_results": [],
        "url_results": [],
        "citations": [],
        "final_answer": None,
        "needs_more_research": False,
        "iteration_count": 1,
        "max_iterations": 3,
        "session_id": "test-session",
        "conversation_history": [],
        "messages": [],
    }
    state.update(overrides)
    return state


# ===========================================================================
# route_after_planner
# ===========================================================================

class TestRouteAfterPlanner:
    """Tests for the fan-out routing logic after the planner node."""

    def test_single_vector_search(self):
        """Single tool 'vector_search' maps to the correct node name."""
        state = _base_state(tools_to_use=["vector_search"])
        result = route_after_planner(state)
        assert result == ["vector_search_node"]

    def test_single_tavily_search(self):
        """Single tool 'tavily_search' maps to 'tavily_search_node'."""
        state = _base_state(tools_to_use=["tavily_search"])
        result = route_after_planner(state)
        assert result == ["tavily_search_node"]

    def test_single_url_fetch(self):
        """Single tool 'url_fetch' maps to 'url_fetch_node'."""
        state = _base_state(tools_to_use=["url_fetch"])
        result = route_after_planner(state)
        assert result == ["url_fetch_node"]

    def test_multiple_tools_fan_out(self):
        """Multiple tools produce a list of all corresponding node names."""
        state = _base_state(tools_to_use=["vector_search", "tavily_search"])
        result = route_after_planner(state)
        assert set(result) == {"vector_search_node", "tavily_search_node"}

    def test_all_three_tools(self):
        """All three valid tools fan out to all three node names."""
        state = _base_state(tools_to_use=["vector_search", "tavily_search", "url_fetch"])
        result = route_after_planner(state)
        assert set(result) == {
            "vector_search_node",
            "tavily_search_node",
            "url_fetch_node",
        }

    def test_empty_tools_falls_back_to_synthesizer(self):
        """If planner chose no tools, route directly to synthesizer."""
        state = _base_state(tools_to_use=[])
        result = route_after_planner(state)
        assert result == ["synthesizer"]

    def test_missing_tools_to_use_key_falls_back_to_synthesizer(self):
        """If 'tools_to_use' key is absent, default to synthesizer."""
        state = _base_state()
        del state["tools_to_use"]
        result = route_after_planner(state)
        assert result == ["synthesizer"]

    def test_unknown_tool_is_filtered_out(self):
        """Unrecognised tool names are silently filtered; valid ones still route."""
        state = _base_state(tools_to_use=["unknown_tool", "vector_search"])
        result = route_after_planner(state)
        assert result == ["vector_search_node"]

    def test_all_unknown_tools_fall_back_to_synthesizer(self):
        """If every listed tool is unknown, fall back to synthesizer."""
        state = _base_state(tools_to_use=["ghost_tool", "phantom_tool"])
        result = route_after_planner(state)
        assert result == ["synthesizer"]

    def test_order_preserved(self):
        """Output order reflects the order of tools_to_use."""
        state = _base_state(tools_to_use=["tavily_search", "vector_search", "url_fetch"])
        result = route_after_planner(state)
        assert result == ["tavily_search_node", "vector_search_node", "url_fetch_node"]

    def test_duplicate_tools_produce_duplicate_nodes(self):
        """Duplicate entries in tools_to_use produce duplicate node names."""
        state = _base_state(tools_to_use=["vector_search", "vector_search"])
        result = route_after_planner(state)
        assert result == ["vector_search_node", "vector_search_node"]

    def test_return_type_is_list(self):
        """route_after_planner always returns a list, never a string."""
        state = _base_state(tools_to_use=["vector_search"])
        result = route_after_planner(state)
        assert isinstance(result, list)


# ===========================================================================
# route_after_grader
# ===========================================================================

class TestRouteAfterGrader:
    """Tests for the conditional routing logic after the grader node."""

    def test_needs_more_research_below_max_iterations_returns_planner(self):
        """When more research is needed and iterations remain, loop to planner."""
        state = _base_state(
            needs_more_research=True,
            iteration_count=1,
            max_iterations=3,
        )
        assert route_after_grader(state) == "planner"

    def test_needs_more_research_at_max_iterations_goes_to_synthesizer(self):
        """When iterations are exhausted, proceed to synthesizer even if more research is needed."""
        state = _base_state(
            needs_more_research=True,
            iteration_count=3,
            max_iterations=3,
        )
        assert route_after_grader(state) == "synthesizer"

    def test_no_more_research_needed_goes_to_synthesizer(self):
        """When research is sufficient, always go to synthesizer."""
        state = _base_state(
            needs_more_research=False,
            iteration_count=1,
            max_iterations=3,
        )
        assert route_after_grader(state) == "synthesizer"

    def test_no_more_research_at_max_iterations_goes_to_synthesizer(self):
        """No more research + max iterations reached => synthesizer."""
        state = _base_state(
            needs_more_research=False,
            iteration_count=3,
            max_iterations=3,
        )
        assert route_after_grader(state) == "synthesizer"

    def test_iteration_count_exceeds_max_goes_to_synthesizer(self):
        """iteration_count strictly greater than max_iterations => synthesizer."""
        state = _base_state(
            needs_more_research=True,
            iteration_count=5,
            max_iterations=3,
        )
        assert route_after_grader(state) == "synthesizer"

    def test_missing_needs_more_research_defaults_to_synthesizer(self):
        """Absent 'needs_more_research' key is falsy; route to synthesizer."""
        state = _base_state(iteration_count=1, max_iterations=3)
        del state["needs_more_research"]
        assert route_after_grader(state) == "synthesizer"

    def test_missing_iteration_count_defaults_to_one(self):
        """
        Absent 'iteration_count' evaluates as 0 via .get default.
        The condition checks count < max, so 0 < 3 => planner when research needed.
        """
        state = _base_state(needs_more_research=True, max_iterations=3)
        del state["iteration_count"]
        # get("iteration_count", 1) => 1; 1 < 3 => True => planner
        assert route_after_grader(state) == "planner"

    def test_missing_max_iterations_defaults_to_three(self):
        """Absent 'max_iterations' defaults to 3 inside route_after_grader."""
        state = _base_state(needs_more_research=True, iteration_count=2)
        del state["max_iterations"]
        # get("max_iterations", 3) => 3; 2 < 3 => planner
        assert route_after_grader(state) == "planner"

    def test_return_type_is_string(self):
        """route_after_grader always returns a single string node name."""
        state = _base_state(needs_more_research=False)
        result = route_after_grader(state)
        assert isinstance(result, str)

    def test_boundary_iteration_count_one_less_than_max(self):
        """iteration_count == max_iterations - 1 should still loop to planner."""
        state = _base_state(
            needs_more_research=True,
            iteration_count=2,
            max_iterations=3,
        )
        assert route_after_grader(state) == "planner"

    def test_boundary_iteration_count_equals_max(self):
        """iteration_count == max_iterations means condition is false, go to synthesizer."""
        state = _base_state(
            needs_more_research=True,
            iteration_count=3,
            max_iterations=3,
        )
        assert route_after_grader(state) == "synthesizer"

    def test_max_iterations_one_always_goes_to_synthesizer(self):
        """With max_iterations=1, a single iteration exhausts the budget."""
        state = _base_state(
            needs_more_research=True,
            iteration_count=1,
            max_iterations=1,
        )
        assert route_after_grader(state) == "synthesizer"
