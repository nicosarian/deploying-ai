"""
llm_client.py
-------------
Every LLM call in this project goes through this one file. That is
deliberate: it is the single seam you need to touch to match the course's
environment.

>>> ACTION NEEDED <<<
This course routes the OpenAI SDK through a custom `get_client()` that
points at an AWS API Gateway proxy (the same pattern used in the RAG /
agents notebooks earlier in the certificate). I don't have that exact
function, so `get_client()` below is a plain, runnable stand-in — replace
its body with your course's version and nothing else in the project needs
to change. Everything downstream (services, guardrails, memory, app.py)
only ever calls `chat_completion()` and `summarize_turns()` from this
file.
"""

import json
import os

from openai import OpenAI  # replace this import too, if your course's
                            # get_client() wraps a different SDK object

from dotenv import load_dotenv
load_dotenv()

USE_GATEWAY = (os.getenv('USE_GATEWAY', 'TRUE').upper() == 'TRUE')

def get_client(use_gateway: bool = USE_GATEWAY) -> OpenAI:
    """Course get_client() from 05_src/utils/clients.py: authenticates
    against the DSI AWS API Gateway proxy via an x-api-key header.
    Set USE_GATEWAY=FALSE in .env to fall back to a plain OpenAI client
    (requires a real sk- key in OPENAI_API_KEY)."""
    if use_gateway:
        client = OpenAI(base_url='https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1',
                    api_key='any value',
                    default_headers={"x-api-key": os.getenv('API_GATEWAY_KEY')})
    else:
        client = OpenAI()
    return client


MODEL_NAME = os.environ.get("CHAT_MODEL", "gpt-4o-mini")


def chat_completion(messages, tools=None, tool_choice="auto", temperature=0.7):
    """Thin wrapper around client.chat.completions.create(). Kept as a
    single function so every caller (the main chat loop and the memory
    summarizer) shares one place to add retries, logging, etc."""
    client = get_client()
    kwargs = dict(model=MODEL_NAME, messages=messages, temperature=temperature)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    return client.chat.completions.create(**kwargs)


def summarize_turns(previous_summary, turns):
    """Condense a list of {"role", "content"} turns (plus any existing
    running summary) into a short paragraph. Used by memory.py to keep
    the context window bounded on long conversations."""
    transcript = "\n".join(
        f"{t.get('role', '?')}: {t.get('content', '')}"
        for t in turns
        if t.get("role") in ("user", "assistant") and t.get("content")
    )
    prompt = (
        "Condense the following older portion of a conversation into at "
        "most 4 sentences of plain prose. Preserve any concrete facts, "
        "names, numbers, or stated preferences the assistant should "
        "still remember; drop small talk. If a previous summary is "
        "given, merge it in rather than discarding it.\n\n"
        f"Previous summary: {previous_summary or '(none)'}\n\n"
        f"Older turns:\n{transcript}"
    )
    client = get_client()
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def parse_tool_arguments(raw_arguments: str) -> dict:
    """Defensive JSON parsing for tool-call arguments — models
    occasionally emit malformed JSON; fail soft rather than crash the
    chat turn."""
    try:
        return json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return {}
