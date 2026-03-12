import json
import re
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.config import settings
from app.agent.state import AgentState, SourceCitation
from app.agent.prompts import PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT
from app.tools.vector_search import vector_search
from app.tools.tavily_search import tavily_search
from app.tools.url_fetch import url_fetch


def _get_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.CLAUDE_MODEL,
        anthropic_api_key=settings.ANTHROPIC_API_KEY,
        anthropic_api_url=settings.ANTHROPIC_BASE_URL,
        max_tokens=settings.CLAUDE_MAX_TOKENS,
        temperature=settings.CLAUDE_TEMPERATURE,
    )


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["## Prior conversation\n"]
    for turn in history[-settings.MAX_CONVERSATION_HISTORY_TURNS * 2:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"**{role}**: {turn['content']}\n")
    return "\n".join(lines)


async def planner(state: AgentState) -> dict:
    """
    Call Claude to decide which tools to invoke and optionally rewrite the query.
    Returns structured JSON: {tools_to_use, refined_query}.
    """
    llm = _get_llm()
    history_text = _format_history(state.get("conversation_history", []))
    user_content = f"{history_text}\n\n## Current question\n{state['query']}"

    response = await llm.ainvoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    # Parse JSON from response
    raw = response.content.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: use both retrieval methods
        plan = {"tools_to_use": ["vector_search", "tavily_search"], "refined_query": None}

    return {
        "tools_to_use": plan.get("tools_to_use", []),
        "refined_query": plan.get("refined_query") or None,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "tools_used": [],
        "vector_results": [],
        "web_results": [],
        "url_results": [],
    }


async def vector_search_node(state: AgentState) -> dict:
    query = state.get("refined_query") or state["query"]
    results = await vector_search.ainvoke({"query": query, "k": settings.VECTOR_SEARCH_K})
    return {
        "vector_results": results if isinstance(results, list) else [],
        "tools_used": state.get("tools_used", []) + ["vector_search"],
    }


async def tavily_search_node(state: AgentState) -> dict:
    query = state.get("refined_query") or state["query"]
    results = await tavily_search.ainvoke({"query": query, "max_results": 5})
    return {
        "web_results": results if isinstance(results, list) else [],
        "tools_used": state.get("tools_used", []) + ["tavily_search"],
    }


async def url_fetch_node(state: AgentState) -> dict:
    """Fetch URLs found in the query or in web results."""
    import re as _re
    query = state["query"]
    urls = _re.findall(r"https?://[^\s]+", query)

    url_results = []
    for url in urls[:3]:  # cap at 3 URLs per turn
        result = await url_fetch.ainvoke({"url": url, "max_chars": 8000})
        if result.get("content"):
            url_results.append(result)

    return {
        "url_results": url_results,
        "tools_used": state.get("tools_used", []) + ["url_fetch"],
    }


async def grader(state: AgentState) -> dict:
    """
    Evaluate whether retrieved context is sufficient.
    Simple heuristic: if any source returned results, we have enough context.
    """
    total = (
        len(state.get("vector_results") or [])
        + len(state.get("web_results") or [])
        + len(state.get("url_results") or [])
    )
    needs_more = total == 0 and state.get("iteration_count", 1) < state.get("max_iterations", 3)
    return {"needs_more_research": needs_more}


def _build_context_block(state: AgentState) -> tuple[str, list[SourceCitation]]:
    """Format all retrieved results into a context string with numbered citations."""
    parts = []
    citations: list[SourceCitation] = []
    idx = 1

    for item in state.get("vector_results") or []:
        meta = item.get("metadata", {})
        title = meta.get("original_filename") or meta.get("source", "Uploaded Document")
        citation: SourceCitation = {
            "source_type": "vector",
            "title": title,
            "url": None,
            "chunk_id": meta.get("chunk_id"),
            "score": item.get("score"),
            "excerpt": item["content"][:200],
        }
        citations.append(citation)
        parts.append(f"[Source {idx}] ({title})\n{item['content']}")
        idx += 1

    for item in state.get("web_results") or []:
        title = item.get("title", "Web Result")
        url = item.get("url", "")
        citation: SourceCitation = {
            "source_type": "web_search",
            "title": title,
            "url": url,
            "chunk_id": None,
            "score": item.get("score"),
            "excerpt": item.get("content", item.get("snippet", ""))[:200],
        }
        citations.append(citation)
        content = item.get("content", item.get("snippet", ""))
        parts.append(f"[Source {idx}] ({title} — {url})\n{content}")
        idx += 1

    for item in state.get("url_results") or []:
        url = item.get("url", "")
        citation: SourceCitation = {
            "source_type": "url_fetch",
            "title": url,
            "url": url,
            "chunk_id": None,
            "score": None,
            "excerpt": item.get("content", "")[:200],
        }
        citations.append(citation)
        parts.append(f"[Source {idx}] ({url})\n{item.get('content', '')}")
        idx += 1

    return "\n\n---\n\n".join(parts), citations


async def synthesizer(state: AgentState) -> dict:
    """Combine all retrieved context and generate a cited answer."""
    llm = _get_llm()
    context_text, citations = _build_context_block(state)

    if context_text:
        user_content = (
            f"## Retrieved Context\n\n{context_text}\n\n"
            f"## Question\n{state['query']}"
        )
    else:
        user_content = (
            f"No external context was retrieved. Answer from your own knowledge if possible.\n\n"
            f"## Question\n{state['query']}"
        )

    response = await llm.ainvoke([
        SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ])

    return {
        "final_answer": response.content,
        "citations": citations,
    }


async def responder(state: AgentState) -> dict:
    """Append the final answer to messages and save conversation history."""
    from app.memory.conversation import save_conversation_turn

    answer = state.get("final_answer", "I was unable to find an answer.")
    save_conversation_turn(
        session_id=state["session_id"],
        query=state["query"],
        answer=answer,
        citations=state.get("citations", []),
    )

    return {
        "messages": [AIMessage(content=answer)],
    }
