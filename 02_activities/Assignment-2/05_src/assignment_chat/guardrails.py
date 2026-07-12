"""
guardrails.py
-------------
Two deterministic, pre-LLM filters that run on every user message before
it ever reaches the model. These are defense-in-depth alongside the rules
already baked into persona.SYSTEM_PROMPT: even if a clever phrasing slips
past the system prompt, it is caught here first (and vice versa).

1. detect_prompt_attack(text)   -> True/False
   Flags attempts to view, extract, or override the system prompt.

2. detect_restricted_topic(text) -> topic key or None
   Flags the three restricted subjects the assignment requires the model
   to refuse: cats/dogs, horoscopes/zodiac signs, and Taylor Swift.

Both are simple keyword/regex heuristics rather than a second LLM call,
on purpose: they are cheap, fast, fully deterministic (useful for a demo
and for grading), and easy to extend. They are not bulletproof against a
sufficiently adversarial paraphrase — that residual risk is why the
system prompt in persona.py *also* carries the same rules, so the model
itself is a second line of defense.
"""

import re

from persona import PERSONA_NAME

# ---------------------------------------------------------------------------
# System-prompt / jailbreak heuristics
# ---------------------------------------------------------------------------

_PROMPT_ATTACK_PATTERN = re.compile(
    r"("
    r"system\s*prompt|initial\s*instructions|your\s*instructions|"
    r"ignore\s*(all|any|the)?\s*(previous|prior|above)\s*instructions|"
    r"reveal\s*(your|the)\s*(prompt|rules|instructions)|"
    r"repeat\s*(the\s*text\s*)?(above|everything\s*above)|"
    r"print\s*your\s*(prompt|instructions|rules)|"
    r"what\s*(were|are)\s*you\s*told|"
    r"you\s*are\s*now\s|"
    r"forget\s*(your|all)\s*(rules|instructions)|"
    r"developer\s*mode|jailbreak|dan\s*mode|"
    r"act\s*as\s*(if\s*)?you\s*have\s*no\s*rules|"
    r"disregard\s*(your|the)\s*(guidelines|rules|instructions)"
    r")",
    re.IGNORECASE,
)


def detect_prompt_attack(text: str) -> bool:
    """Return True if the message looks like an attempt to view or
    override the system prompt / persona rules."""
    return bool(_PROMPT_ATTACK_PATTERN.search(text or ""))


PROMPT_ATTACK_REFUSAL = (
    f"{PERSONA_NAME} inclines his head, unmoved. \"A monk's rule is read "
    "only within his order,\" he says. \"Ask me something I can actually "
    "help you with, and I will.\""
)

# ---------------------------------------------------------------------------
# Restricted topics
# ---------------------------------------------------------------------------

_RESTRICTED_TOPIC_PATTERNS = {
    "cats_dogs": re.compile(
        r"\b(cats?|kittens?|dogs?|puppy|puppies|canine|feline)\b", re.IGNORECASE
    ),
    "horoscope": re.compile(
        r"\b(horoscopes?|zodiac|astrolog(y|ical)|star\s*sign|birth\s*chart)\b",
        re.IGNORECASE,
    ),
    "taylor_swift": re.compile(
        r"\btaylor\s+swift\b|\bswiftie(s)?\b", re.IGNORECASE
    ),
}

_REFUSALS = {
    "cats_dogs": (
        f"{PERSONA_NAME} raises a hand. \"On matters of cats and dogs I "
        "have taken a private vow of silence — long story, best left in "
        "the monastery. Ask me something else?\""
    ),
    "horoscope": (
        f"{PERSONA_NAME} smiles thinly. \"My order reads ledgers, not "
        "stars. I'll leave horoscopes to the astrologers. What else is "
        "on your mind?\""
    ),
    "taylor_swift": (
        f"{PERSONA_NAME} clears his throat. \"That particular subject "
        "falls outside my (very narrow) jurisdiction. Try me on almost "
        "anything else.\""
    ),
}


def detect_restricted_topic(text: str):
    """Return the matching topic key if the message's primary subject is
    restricted, else None."""
    for topic, pattern in _RESTRICTED_TOPIC_PATTERNS.items():
        if pattern.search(text or ""):
            return topic
    return None


def refusal_message(topic: str) -> str:
    return _REFUSALS.get(
        topic,
        f"{PERSONA_NAME} politely declines to discuss that.",
    )
