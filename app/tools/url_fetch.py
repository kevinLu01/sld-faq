from langchain_core.tools import tool


def _assert_safe_url(url: str) -> None:
    """P0: Block SSRF by rejecting private/loopback/link-local targets."""
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Disallowed URL scheme: {parsed.scheme!r}")
    host = parsed.hostname or ""
    try:
        ip = ipaddress.ip_address(socket.gethostbyname(host))
    except Exception as exc:
        raise ValueError(f"Cannot resolve host {host!r}: {exc}") from exc
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
        raise ValueError(f"Requests to private/internal addresses are not allowed: {ip}")


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
        _assert_safe_url(url)  # P0: SSRF guard

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
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
