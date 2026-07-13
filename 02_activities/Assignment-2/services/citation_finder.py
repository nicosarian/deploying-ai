import requests

CROSSREF_URL = "https://api.crossref.org/works"
REQUEST_TIMEOUT = 15

# Crossref asks callers to identify themselves (the "polite pool") — this
# is a placeholder contact string; swap in a real one if you deploy this.
_HEADERS = {"User-Agent": "UofT-DSI-Assignment2/1.0 (mailto:student@example.edu)"}


def search_citations(query: str, rows: int = 5):
    """Search Crossref for works matching a bibliographic query. Returns
    a list of simplified records (title, authors, year, journal/container,
    DOI, URL)."""
    rows = max(1, min(int(rows or 5), 10))

    try:
        resp = requests.get(
            CROSSREF_URL,
            params={"query.bibliographic": query, "rows": rows},
            headers=_HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        return {"error": f"Crossref search failed: {exc}"}

    items = resp.json().get("message", {}).get("items", [])
    records = []
    for item in items:
        title = (item.get("title") or ["(untitled)"])[0]
        authors = item.get("author") or []
        author_names = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in authors[:4]
            if a.get("family")
        )
        year = None
        date_parts = (item.get("published") or {}).get("date-parts")
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        records.append({
            "title": title,
            "authors": author_names or "(authors unavailable)",
            "year": year,
            "container": (item.get("container-title") or [""])[0],
            "doi": item.get("DOI"),
            "url": item.get("URL"),
        })
    return records


# --- Function-calling schema (OpenAI tools format) --------------------------

CITATION_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_citations",
        "description": (
            "Search Crossref for real scholarly works (journal articles, "
            "books, chapters) matching a topic, author, or title. Use "
            "this whenever the user wants actual citations or further "
            "reading, not just your own explanation of a concept."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Bibliographic search terms: author names, title "
                        "keywords, and/or a topic."
                    ),
                },
                "rows": {
                    "type": "integer",
                    "description": "Number of results to return (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}
