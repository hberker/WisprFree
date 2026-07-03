from wisperfree.llm.prompts import build_cleanup_system_prompt


def test_base_rules_present():
    prompt = build_cleanup_system_prompt()
    assert "filler words" in prompt
    assert "self-corrections" in prompt
    assert "meet Wednesday" in prompt  # the canonical backtracking example
    assert "Output ONLY the rewritten text" in prompt


def test_tone_and_app_injected():
    prompt = build_cleanup_system_prompt(tone="casual", app_name="Slack")
    assert "Slack" in prompt
    assert "casual chat message" in prompt

    formal = build_cleanup_system_prompt(tone="formal")
    assert "professional writing" in formal

    technical = build_cleanup_system_prompt(tone="technical")
    assert "code identifiers" in technical


def test_unknown_tone_falls_back_to_neutral():
    assert "neutral" in build_cleanup_system_prompt(tone="sarcastic")


def test_dictionary_terms_injected():
    prompt = build_cleanup_system_prompt(
        dictionary_terms=["Hosni", "WisperFree", "PEFT"]
    )
    assert "Hosni, WisperFree, PEFT" in prompt


def test_extra_instructions_appended():
    prompt = build_cleanup_system_prompt(extra_instructions="never use emojis")
    assert "never use emojis" in prompt
