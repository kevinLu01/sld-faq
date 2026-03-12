from langchain_core.tools import tool


@tool
async def url_fetch(url: str, max_chars: int = 8000) -> dict:
    """
    Fetch and extract clean text content from a given URL.
    Use this when the user provides a specific URL or when a search result
    warrants deeper reading of the full page.
    """
    import httpx
    from bs4 import BeautifulSoup

    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(
                url, headers={"User-Agent": "ResearchAgent/1.0"}
            )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)[:max_chars]

        return {
            "url": url,
            "content": text,
            "status_code": response.status_code,
            "error": None,
        }
    except Exception as e:
        return {"url": url, "content": "", "status_code": 0, "error": str(e)}
