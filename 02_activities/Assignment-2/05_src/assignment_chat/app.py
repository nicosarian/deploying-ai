"""
app.py
------
Entry point: `python app.py` launches the Gradio chat interface.

Architecture in one paragraph: every user message first passes through
two deterministic guardrail checks (guardrails.py). If it clears those,
it's added to a ConversationMemory (memory.py) and sent to the model
along with three tool schemas — one per service. The model decides for
itself whether answering requires calling search_met_artwork (Service 1),
search_theology_corpus (Service 2), search_citations (Service 3), some
combination, or none at all. Tool results are fed back to the model,
which produces the final in-persona reply. This is a single small
agent loop, not three separate hardcoded buttons — the "which service do
I need" decision is exactly what we're delegating to function calling.
"""

import inspect
import json

import gradio as gr

from guardrails import detect_prompt_attack, detect_restricted_topic, refusal_message, PROMPT_ATTACK_REFUSAL
from llm_client import chat_completion, summarize_turns, parse_tool_arguments
from memory import ConversationMemory
from persona import PERSONA_NAME, SYSTEM_PROMPT
from services.art_lookup import search_met, ART_TOOL_SCHEMA
from services.citation_finder import search_citations, CITATION_TOOL_SCHEMA
from services.semantic_search import semantic_query, SEMANTIC_TOOL_SCHEMA

ALL_TOOLS = [ART_TOOL_SCHEMA, SEMANTIC_TOOL_SCHEMA, CITATION_TOOL_SCHEMA]

TOOL_IMPLEMENTATIONS = {
    "search_met_artwork": search_met,
    "search_theology_corpus": semantic_query,
    "search_citations": search_citations,
}

MAX_TOOL_HOPS = 3          # safety cap on tool-call round trips per turn
MAX_TOOL_RESULT_CHARS = 6000  # keep tool output from blowing up the context


def run_tool_calls(messages, tool_calls):
    """Execute each requested tool call and append its result as a
    tool-role message. Mutates and returns `messages`."""
    for call in tool_calls:
        name = call.function.name
        args = parse_tool_arguments(call.function.arguments)
        implementation = TOOL_IMPLEMENTATIONS.get(name)

        if implementation is None:
            result = {"error": f"Unknown tool '{name}'"}
        else:
            try:
                result = implementation(**args)
            except Exception as exc:  # a live external API can fail in many ways
                result = {"error": f"{name} failed: {exc}"}

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result)[:MAX_TOOL_RESULT_CHARS],
        })
    return messages


def get_agent_reply(memory: ConversationMemory) -> str:
    """Run the tool-calling loop until the model produces a final text
    reply (or the hop limit is hit), and return that reply."""
    messages = memory.to_messages()
    hops = 0

    while True:
        completion = chat_completion(messages, tools=ALL_TOOLS)
        message = completion.choices[0].message

        if getattr(message, "tool_calls", None) and hops < MAX_TOOL_HOPS:
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": message.tool_calls,
            })
            run_tool_calls(messages, message.tool_calls)
            hops += 1
            continue

        return message.content or "..."


def respond(user_message, chat_history, memory_state):
    chat_history = chat_history or []
    user_message = (user_message or "").strip()
    if not user_message:
        return chat_history, memory_state, ""

    if memory_state is None:
        memory_state = ConversationMemory(SYSTEM_PROMPT)

    # --- Guardrails: checked before the model ever sees the message -----
    if detect_prompt_attack(user_message):
        chat_history = chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": PROMPT_ATTACK_REFUSAL},
        ]
        return chat_history, memory_state, ""

    topic = detect_restricted_topic(user_message)
    if topic:
        chat_history = chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": refusal_message(topic)},
        ]
        return chat_history, memory_state, ""

    # --- Normal turn ------------------------------------------------------
    memory_state.add_user(user_message)
    reply = get_agent_reply(memory_state)
    memory_state.add_assistant(reply)
    memory_state.maybe_compact(summarize_turns)

    chat_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return chat_history, memory_state, ""


def reset():
    return [], None, ""


with gr.Blocks(title=f"{PERSONA_NAME}") as demo:
    gr.Markdown(
        f"## \U0001F56F\uFE0F {PERSONA_NAME}\n"
        "*An almoner, reassigned. Ask about sacred art, the economics of "
        "renunciation, or scholarly sources on sacrifice and luxury.*"
    )
    # Gradio 5 needs type="messages" to accept role/content dicts;
    # Gradio 6 removed the kwarg (messages is the only format).
    _chatbot_kwargs = dict(height=480, show_label=False)
    if "type" in inspect.signature(gr.Chatbot.__init__).parameters:
        _chatbot_kwargs["type"] = "messages"
    chatbot = gr.Chatbot(**_chatbot_kwargs)
    memory_state = gr.State(None)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Speak, and I shall weigh your words...",
            show_label=False,
            scale=5,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")

    clear_btn = gr.Button("Reset conversation")

    send_btn.click(respond, [msg, chatbot, memory_state], [chatbot, memory_state, msg])
    msg.submit(respond, [msg, chatbot, memory_state], [chatbot, memory_state, msg])
    clear_btn.click(reset, None, [chatbot, memory_state, msg])

if __name__ == "__main__":
    demo.launch(show_error=True)
