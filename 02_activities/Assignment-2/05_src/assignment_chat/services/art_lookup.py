"""
services/art_lookup.py
-----------------------
SERVICE 1: API Calls (assignment requirement).

Backend: The Metropolitan Museum of Art Open Access Collection API.
  - Free, no API key or registration required.
  - Base URL: https://collectionapi.metmuseum.org/public/collection/v1
  - We call /search?q=... to get matching object IDs, then /objects/{id}
    for each one's full record.
  - Docs: https://metmuseum.github.io/

Per the assignment, raw API output must never be shown to the user
verbatim. This module only returns structured data; the chat loop in
app.py hands that structured data to the model, which narrates it in
persona instead of dumping JSON. This module itself does no rephrasing —
that separation keeps the "fetch data" and "present data" concerns apart.
"""

import requests

BASE_URL = "https://collectionapi.metmuseum.org/public/collection/v1"
REQUEST_TIMEOUT = 15


def search_met(query: str, limit: int = 4):
    """Search the Met's collection for objects matching `query` and
    return a short list of simplified records (title, artist, date,
    medium, culture, department, credit line, image URL, public-domain
    flag). Returns [] if nothing is found or the API is unreachable."""
    limit = max(1, min(int(limit or 4), 8))

    try:
        search_resp = requests.get(
            f"{BASE_URL}/search",
            params={"q": query, "hasImages": "true"},
            timeout=REQUEST_TIMEOUT,
        )
        search_resp.raise_for_status()
    except requests.RequestException as exc:
        return {"error": f"Met API search failed: {exc}"}

    object_ids = (search_resp.json().get("objectIDs") or [])[:limit]
    if not object_ids:
        return []

    records = []
    for object_id in object_ids:
        try:
            obj_resp = requests.get(
                f"{BASE_URL}/objects/{object_id}", timeout=REQUEST_TIMEOUT
            )
        except requests.RequestException:
            continue
        if obj_resp.status_code != 200:
            continue
        obj = obj_resp.json()
        records.append({
            "title": obj.get("title") or "Untitled",
            "artist": obj.get("artistDisplayName") or "Unknown maker",
            "date": obj.get("objectDate") or "date unknown",
            "culture": obj.get("culture") or "",
            "medium": obj.get("medium") or "",
            "department": obj.get("department") or "",
            "credit_line": obj.get("creditLine") or "",
            "image_url": obj.get("primaryImage") or "",
            "object_page": obj.get("objectURL") or "",
            "is_public_domain": bool(obj.get("isPublicDomain")),
        })
    return records


# --- Function-calling schema (OpenAI tools format) --------------------------

ART_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_met_artwork",
        "description": (
            "Search the Metropolitan Museum of Art's open collection for "
            "artworks matching a topic, deity, culture, ritual object, or "
            "theme (e.g. 'bodhisattva', 'reliquary', 'ex-voto offering', "
            "'Madonna and Child'). Returns metadata (not images rendered "
            "inline) for a handful of matching works."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search terms describing the artwork or theme.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return (default 4, max 8).",
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    },
}
