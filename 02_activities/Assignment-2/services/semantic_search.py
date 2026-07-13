"""
services/semantic_search.py
----------------------------
SERVICE 2: Semantic Query (assignment requirement).

Backend: a ChromaDB PersistentClient (file-persisted, no Docker) reading
a collection built by ../build_index.py from the .txt files in
../data/corpus/. See that file, and the README, for how the embeddings
are produced and why.

This module only performs retrieval and returns raw hits (text + source
+ distance). Answer synthesis happens in app.py, where the model is
given these hits as tool output and asked to answer in persona, citing
which note(s) it drew on.
"""

import os

import chromadb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(_THIS_DIR, "..", "data", "chroma_store")
COLLECTION_NAME = "economic_theology_corpus"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_or_create_collection(COLLECTION_NAME)
    return _collection


def semantic_query(query: str, n_results: int = 3):
    """Return up to n_results passages from the corpus most semantically
    similar to `query`, each with its source filename and a distance
    score (lower = more similar)."""
    collection = _get_collection()
    if collection.count() == 0:
        return {
            "error": (
                "The corpus index is empty. Run `python data/build_index.py` "
                "once before using this service."
            )
        }

    n_results = max(1, min(int(n_results or 3), 8))
    results = collection.query(query_texts=[query], n_results=n_results)

    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append({
            "text": doc,
            "source": (meta or {}).get("source", "unknown"),
            "distance": round(float(dist), 4),
        })
    return hits


# --- Function-calling schema (OpenAI tools format) --------------------------

SEMANTIC_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_theology_corpus",
        "description": (
            "Semantic search over a curated corpus of short notes on "
            "asceticism, gift economies, the Protestant ethic, "
            "conspicuous consumption, biblical teachings on wealth, "
            "Bataille's concept of expenditure, and theories of sacrifice "
            "and luxury. Use for conceptual questions about wealth, "
            "renunciation, sacrifice, or expenditure, not for looking up "
            "artworks or scholarly citations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The concept or question to search for.",
                },
                "n_results": {
                    "type": "integer",
                    "description": "Number of passages to retrieve (default 3, max 8).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}
