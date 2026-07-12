"""
data/build_index.py
--------------------
Builds (or rebuilds) the persistent Chroma store used by Service 2 from
the plain-text files in data/corpus/.

Run once, from the assignment_chat/ directory:

    python data/build_index.py

Embedding process (for the README's required description):
  - Each .txt file in data/corpus/ is treated as a single document (no
    chunking) — they are short, single-topic notes (~200 words each)
    written specifically for this assignment, so splitting them further
    would only fragment a single idea.
  - We use ChromaDB's default embedding function (a local sentence-
    transformers model, all-MiniLM-L6-v2, bundled with chromadb) rather
    than an API-based embedding model. This keeps Service 2 fully
    self-contained: no API key or network call is required at query
    time, only on the very first run while the model weights are
    downloaded and cached locally.
  - Metadata stored per document: just its source filename, which lets
    the chat client cite which note an answer drew on.
  - The resulting store lives in data/chroma_store/ and is small enough
    to commit directly to the repo, so graders do not need to re-run
    this script — though it is safe to.
"""

import glob
import os

import chromadb

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_DIR = os.path.join(_THIS_DIR, "corpus")
CHROMA_DIR = os.path.join(_THIS_DIR, "chroma_store")
COLLECTION_NAME = "economic_theology_corpus"


def build():
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Idempotent: clear out anything already indexed before re-adding, so
    # re-running this script after editing the corpus doesn't leave stale
    # or duplicate entries behind.
    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(ids=existing["ids"])

    paths = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    if not paths:
        print(f"No .txt files found in {CORPUS_DIR} — nothing to index.")
        return

    ids, documents, metadatas = [], [], []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            continue
        ids.append(f"doc_{i:03d}")
        documents.append(text)
        metadatas.append({"source": os.path.basename(path)})

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(documents)} documents into collection "
          f"'{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    build()
