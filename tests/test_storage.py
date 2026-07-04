import pytest


def test_dictionary_crud(dictionary):
    dictionary.add("Kubernetes")
    entry = dictionary.add("Anthropic", hint="AI company")
    assert entry.hint == "AI company"
    assert [e.term for e in dictionary.list()] == ["Anthropic", "Kubernetes"]
    assert dictionary.remove("kubernetes") is True  # case-insensitive
    assert [e.term for e in dictionary.list()] == ["Anthropic"]
    assert dictionary.remove("missing") is False


def test_dictionary_upsert_updates_hint(dictionary):
    dictionary.add("PEFT")
    dictionary.add("PEFT", hint="parameter-efficient fine-tuning")
    entries = dictionary.list()
    assert len(entries) == 1
    assert entries[0].hint == "parameter-efficient fine-tuning"


def test_dictionary_rejects_empty(dictionary):
    with pytest.raises(ValueError):
        dictionary.add("   ")


def test_initial_prompt_biasing(dictionary):
    assert dictionary.initial_prompt() == ""
    dictionary.add("Hosni")
    dictionary.add("WisperFree")
    prompt = dictionary.initial_prompt()
    assert prompt.startswith("Glossary:")
    assert "Hosni" in prompt and "WisperFree" in prompt


def test_tone_profile_defaults_seeded(tone_profiles):
    patterns = {p.app_pattern for p in tone_profiles.list()}
    assert "slack" in patterns and "mail" in patterns


def test_tone_profile_match(tone_profiles):
    assert tone_profiles.match("Slack — #general").tone == "casual"
    assert tone_profiles.match("Microsoft Outlook").tone == "formal"
    assert tone_profiles.match("Some Unknown App").tone == "neutral"
    assert tone_profiles.match(None, default_tone="formal").tone == "formal"


def test_tone_profile_longest_pattern_wins(tone_profiles):
    tone_profiles.upsert("code", "technical")
    tone_profiles.upsert("code — review", "formal")
    assert tone_profiles.match("VS Code — review window").tone == "formal"


def test_tone_profile_upsert_and_remove(tone_profiles):
    profile = tone_profiles.upsert("obsidian", "technical", "keep markdown")
    assert profile.instructions == "keep markdown"
    with pytest.raises(ValueError):
        tone_profiles.upsert("x", "shouty")
    assert tone_profiles.remove(profile.id) is True


def test_corrections_log_and_count(corrections):
    corrections.add("helo world", "Hello, world!", app_name="Mail")
    assert corrections.count() == 1
    assert corrections.count(untrained_only=True) == 1
    pair = corrections.training_pairs()[0]
    corrections.mark_trained([pair.id])
    assert corrections.count(untrained_only=True) == 0


def test_suggestions_from_repeated_corrections(corrections, dictionary):
    # User keeps fixing "cubernetis" -> "Kubernetes"
    for i in range(3):
        corrections.add(f"deploy to cubernetis {i}", f"Deploy to Kubernetes {i}")
    suggestions = corrections.suggestions(dictionary.terms())
    assert [s.term for s in suggestions] == ["Kubernetes"]
    assert suggestions[0].count == 3

    # Already in the dictionary -> not suggested again
    dictionary.add("Kubernetes")
    assert corrections.suggestions(dictionary.terms()) == []


def test_suggestions_respect_dismissal(corrections, dictionary):
    for i in range(3):
        corrections.add(f"see amiga {i}", f"See Amical {i}")
    assert [s.term for s in corrections.suggestions([])] == ["Amical"]
    corrections.dismiss_suggestion("Amical")
    assert corrections.suggestions([]) == []


def test_lowercase_words_need_higher_threshold(corrections):
    for i in range(3):
        corrections.add(f"raw text {i}", f"raw text plus banana {i}")
    # 3 occurrences of a lowercase word < 2*threshold -> no suggestion
    assert corrections.suggestions([]) == []
