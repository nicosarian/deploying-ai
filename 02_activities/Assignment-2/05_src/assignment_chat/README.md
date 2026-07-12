# Brother Largesse — Assignment 2

A conversational AI system built around a single persona, **Brother
Largesse**, a fictional monastic almoner reassigned to run an AI help
desk. He treats every question as a small economy — something is always
being spent, saved, offered, or renounced — which is also the thread
that ties the three otherwise-unrelated backend services into one
coherent chat client instead of three bolted-on demos.

## Architecture

One small agent loop, not three hardcoded buttons. Every service is
exposed to the model as a callable tool; the model itself decides, per
user message, which tool(s) (if any) a good answer requires, calls them,
and then narrates the results in character. This is deliberate: the
assignment's Service 3 requirement is function calling, and using
function calling as the connective tissue for *all three* services is a
more honest demonstration of it than only using it for one isolated
feature.

```
user message
   -> guardrails.py (two deterministic pre-checks)
   -> memory.py (adds to running conversation state)
   -> llm_client.chat_completion(messages, tools=[...])
        -> model may call: search_met_artwork | search_theology_corpus | search_citations
        -> tool result fed back to the model
   -> final in-persona reply
   -> memory.py (may trigger summarization if the conversation is long)
```

## The three services

### Service 1 — API calls: sacred/ritual art lookup
`services/art_lookup.py` calls the **Metropolitan Museum of Art Open
Access Collection API** (`collectionapi.metmuseum.org`, free, no API key).
Given a topic, it searches for matching objects and fetches full records
(title, artist, date, culture, medium, credit line, image URL). The raw
JSON is never shown to the user — it's handed to the model as tool
output, which narrates it in Brother Largesse's voice (e.g. describing a
reliquary the way an almoner would describe an item in a treasury
inventory), satisfying the "must not be provided verbatim" requirement.

### Service 2 — Semantic query: the economic-theology corpus
`services/semantic_search.py` queries a **ChromaDB PersistentClient**
(file-persisted, no Docker) over a small original corpus in
`data/corpus/` — eight short notes (~200 words each) on asceticism, gift
economies (Mauss/potlatch), Weber's Protestant ethic, Veblen's
conspicuous consumption, biblical teachings on wealth, Bataille's concept
of expenditure, theories of sacrifice, and luxury as sacrificial
expenditure.

**Embedding process:** each corpus file is one document (no chunking —
they're short, single-topic notes, so splitting would fragment a single
idea). Embeddings use **ChromaDB's default embedding function**
(`all-MiniLM-L6-v2`, a local sentence-transformers model bundled with
`chromadb`), not an API-based embedding model — this keeps the service
self-contained, with no API key or network call needed at *query* time
(only on the very first run, while the model weights download and
cache locally). Metadata stored per document is just its source
filename, so answers can cite which note they drew on.

**Note on the corpus:** these eight files are original notes written
for this assignment, not excerpts from any single scholar's actual
prose — a deliberate choice to avoid any copyright ambiguity in a
repository that will be public. `data/build_index.py` rebuilds the
index from scratch and is safe to re-run after editing the corpus. If
you want to point this at your actual dissertation material later, drop
new `.txt` files into `data/corpus/` (respecting the 40 MB guidance) and
re-run the build script — nothing else needs to change.

### Service 3 — Function calling: scholarly citation lookup
`services/citation_finder.py` exposes `search_citations` as a callable
tool that queries **Crossref's REST API** (`api.crossref.org`, free, no
key) via `query.bibliographic`, which Crossref documents as intended
specifically for citation lookup. The model decides on its own when a
question calls for real literature (vs. its own explanation) and invokes
the tool, then presents results as a short reading list rather than raw
JSON.

## User interface

`app.py` is a Gradio `Blocks` app (not `ChatInterface`) so that
conversation memory can be managed explicitly via `gr.State` rather than
being reconstructed from Gradio's own chat history each turn — this is
what lets the summarization step (below) persist correctly across turns.

## Memory management

`memory.py` implements a simplified trim-and-summarize strategy (the
pattern referenced in the assignment via LangGraph's short-term memory
docs):
- The system prompt is always pinned and never trimmed.
- The most recent 6 user/assistant turn-pairs are always kept verbatim.
- Once raw history exceeds 10 turn-pairs, everything older than the
  kept window is folded into a single running summary via one extra LLM
  call (`llm_client.summarize_turns`), which **merges** with any
  previous summary rather than discarding it, so it stays bounded
  without losing early context entirely.

Turn-count (rather than token-count) is used as the trigger, on purpose:
it's a coarse but transparent proxy that doesn't require knowing the
exact context window of whatever model sits behind `get_client()`, and
it's easy to demo — just chat for a while and watch it trigger.

## Guardrails

Two deterministic, regex-based pre-checks in `guardrails.py` run on
*every* user message before it reaches the model at all:
1. **System-prompt protection** — flags attempts to view, extract, quote,
   or override the system prompt or persona rules (including common
   jailbreak phrasing), and returns a fixed in-character refusal without
   ever calling the model.
2. **Restricted topics** — flags messages whose primary subject is cats
   or dogs, horoscopes/zodiac signs, or Taylor Swift, and returns a
   persona-appropriate refusal.

This is intentionally defense-in-depth: `persona.py`'s system prompt
*also* carries the same two rules explicitly, spelled out as
non-negotiable, so a phrasing that slips past the regex heuristics still
has to get past the model's own instructions. Neither layer explains its
own detection logic in its refusals, to avoid handing back a map of
exactly what to reword around.

## Setup

```
cd 05_src/assignment_chat
python data/build_index.py   # one-time: builds the Chroma store (needs
                              # internet on first run only, to fetch the
                              # local embedding model)
python app.py
```

### One thing you need to change

`llm_client.py` has a single clearly-marked function, `get_client()`,
that this project uses as its only seam to the LLM backend. It currently
returns a plain `OpenAI()` client. **Replace its body with this course's
`get_client()`** (the one that authenticates against the AWS API Gateway
proxy used elsewhere in the certificate) — nothing else in the project
needs to change, since every other file only calls
`llm_client.chat_completion()` / `llm_client.summarize_turns()`.

## Known limitations / design choices

- The Chroma store ships unbuilt (no pre-built `data/chroma_store/` in
  the repo) so the first run needs one-time internet access to fetch the
  embedding model; after that, Service 2 works fully offline.
- Guardrails are regex heuristics, not a second model call, which is a
  deliberate cost/latency tradeoff — the system prompt is the second
  line of defense against phrasings that slip past them.
- The Met and Crossref services fail soft (return an `error` field
  rather than raising) if the API is unreachable, so a network hiccup on
  one tool doesn't crash the whole turn.
