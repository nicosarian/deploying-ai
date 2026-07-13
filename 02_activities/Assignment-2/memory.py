"""
memory.py
---------
Short-term conversation memory for the chat client.

Strategy (a simplified version of the trim-and-summarize pattern
referenced in the assignment via LangGraph's short-term memory docs):

  - The system prompt is always pinned at position 0; it is never trimmed
    or summarized.
  - The most recent `KEEP_TURNS` user/assistant turn-pairs are always
    kept verbatim, so recent context is never lossy.
  - Once the raw (un-summarized) history grows past `TRIGGER_TURNS`
    turn-pairs, everything older than the kept window is folded into a
    single running summary (one extra system-role message) via an LLM
    call. The summary is re-summarized (merged, not replaced) each time
    this happens, so it keeps growing more compressed rather than
    growing without bound.

This bounds the context window without needing to know the exact token
limit of whatever model the course points at — turn-count is a coarse
but simple and transparent proxy, and easy to demo (you can see it
trigger by just chatting for a while).
"""

KEEP_TURNS = 6        # most recent user+assistant pairs kept verbatim
TRIGGER_TURNS = 10     # raw turn-pairs beyond which older history gets folded


class ConversationMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.summary = None          # condensed memory of older turns, or None
        self.turns = []              # list of {"role": "...", "content": "..."}

    def add_user(self, text: str):
        self.turns.append({"role": "user", "content": text})

    def add_assistant(self, text: str):
        self.turns.append({"role": "assistant", "content": text})

    def _turn_pairs(self) -> int:
        # a rough pair-count; good enough for a coarse trigger
        return len(self.turns) // 2

    def maybe_compact(self, summarizer_fn):
        """Call after each assistant turn. summarizer_fn(previous_summary,
        older_turns) -> new_summary_text, typically llm_client.summarize_turns."""
        if self._turn_pairs() <= TRIGGER_TURNS:
            return
        keep_from = max(0, len(self.turns) - KEEP_TURNS * 2)
        older = self.turns[:keep_from]
        recent = self.turns[keep_from:]
        if not older:
            return
        self.summary = summarizer_fn(self.summary, older)
        self.turns = recent

    def to_messages(self):
        """Build the full message list to send to the model this turn."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Summary of earlier conversation (for context only): {self.summary}",
            })
        messages.extend(self.turns)
        return messages
