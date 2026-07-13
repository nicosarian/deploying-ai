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
