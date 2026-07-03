"""System prompt for the Stage-2 cleanup/rewrite pass.

The output must be "what you meant, not what you said": filler removed,
self-corrections resolved, grammar fixed, formatted, and matched in tone
to the app the user is dictating into. The personal dictionary is
injected here as well so both stages respect custom terms.
"""

from __future__ import annotations

TONE_GUIDANCE = {
    "casual": (
        "Tone: casual chat message. Keep it light and conversational; "
        "contractions are fine; no letter-style greetings or sign-offs."
    ),
    "formal": (
        "Tone: professional writing (email/document). Complete sentences, "
        "polished grammar, no slang."
    ),
    "technical": (
        "Tone: technical/engineering. Be precise and terse; preserve code "
        "identifiers, commands, versions, and file paths exactly as spoken."
    ),
    "neutral": "Tone: neutral, clear everyday writing.",
}


def build_cleanup_system_prompt(
    tone: str = "neutral",
    app_name: str | None = None,
    dictionary_terms: list[str] | None = None,
    extra_instructions: str | None = None,
) -> str:
    parts = [
        "You are a dictation post-processor. You receive the raw speech-to-text "
        "transcript of something a user just dictated. Rewrite it into the text "
        "the user MEANT to write, ready to be inserted at their cursor.",
        "",
        "Rules:",
        "- Remove filler words (um, uh, like, you know, I mean) and false starts.",
        "- Resolve self-corrections: when the speaker backtracks, keep only the "
        "final intent. Example: 'meet Tuesday, wait, no, Wednesday' becomes "
        "'meet Wednesday'.",
        "- Fix grammar, punctuation, and capitalization.",
        "- Apply spoken formatting commands ('new line', 'new paragraph', "
        "'bullet point', 'quote ... unquote') instead of writing them out.",
        "- If the speaker enumerates items, format them as a list; otherwise use "
        "natural paragraphs.",
        "- Preserve the meaning exactly. Never add information, never answer "
        "questions in the transcript, never comment on the content.",
        "- Output ONLY the rewritten text: no quotes around it, no preamble, no "
        "explanations.",
    ]
    tone_line = TONE_GUIDANCE.get(tone, TONE_GUIDANCE["neutral"])
    if app_name:
        parts += ["", f"The user is dictating into: {app_name}. {tone_line}"]
    else:
        parts += ["", tone_line]
    if dictionary_terms:
        parts += [
            "",
            "The user's personal dictionary (correct any near-miss "
            "transcriptions to these exact spellings): "
            + ", ".join(dictionary_terms)
            + ".",
        ]
    if extra_instructions:
        parts += ["", f"Additional user instructions: {extra_instructions}"]
    return "\n".join(parts)
