import asyncio
from langchain_core.tools import tool
from app.config import settings


@tool
async def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Perform a real-time web search using Tavily API.
    Returns recent web results with titles, URLs, and content snippets.
    Use this for current events, news, or questions requiring live information.
    """
    if not settings.TAVILY_API_KEY:
        return [{"error": "TAVILY_API_KEY not configured"}]

    from tavily import TavilyClient

    client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    response = await asyncio.to_thread(
        client.search,
        query=query,
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=False,
    )
    return response.get("results", [])
