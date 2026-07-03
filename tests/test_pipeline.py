import numpy as np

from tests.conftest import FakeASR, FakeLLM, make_pipeline, speech_audio


def test_phase1_raw_injection_when_llm_disabled(config, dictionary, tone_profiles):
    config.llm.enabled = False
    pipeline, injector = make_pipeline(config, dictionary, tone_profiles)
    event = pipeline.process_chunk(speech_audio())
    assert event.raw_text == event.final_text
    assert event.llm_used is False


def test_two_stage_cleanup(config, dictionary, tone_profiles):
    llm = FakeLLM(reply="Meet on Wednesday.")
    pipeline, injector = make_pipeline(config, dictionary, tone_profiles, llm=llm)
    event = pipeline.process_chunk(speech_audio())
    assert event.raw_text == "um so meet on Tuesday no wait Wednesday"
    assert event.final_text == "Meet on Wednesday."
    assert event.llm_used is True
    # LLM got the raw transcript as the user message
    system, user = llm.calls[0]
    assert user == event.raw_text


def test_dictionary_biases_both_stages(config, dictionary, tone_profiles):
    dictionary.add("WisperFree")
    asr, llm = FakeASR(), FakeLLM()
    pipeline, _ = make_pipeline(config, dictionary, tone_profiles, asr=asr, llm=llm)
    pipeline.process_chunk(speech_audio())
    assert "WisperFree" in asr.last_initial_prompt  # Stage 1 initial_prompt
    system, _ = llm.calls[0]
    assert "WisperFree" in system  # Stage 2 system prompt


def test_tone_matched_to_active_app(config, dictionary, tone_profiles):
    llm = FakeLLM()
    pipeline, _ = make_pipeline(
        config, dictionary, tone_profiles, llm=llm, app_name="Slack"
    )
    event = pipeline.process_chunk(speech_audio())
    assert event.tone == "casual"
    system, _ = llm.calls[0]
    assert "Slack" in system and "casual" in system


def test_llm_failure_falls_back_to_raw(config, dictionary, tone_profiles):
    llm = FakeLLM()
    llm.fail = True
    pipeline, _ = make_pipeline(config, dictionary, tone_profiles, llm=llm)
    event = pipeline.process_chunk(speech_audio())
    assert event.final_text == event.raw_text
    assert event.llm_used is False


def test_empty_transcript_produces_no_event(config, dictionary, tone_profiles):
    pipeline, _ = make_pipeline(
        config, dictionary, tone_profiles, asr=FakeASR(text="  ")
    )
    assert pipeline.process_chunk(speech_audio()) is None


def test_worker_loop_injects_text(config, dictionary, tone_profiles):
    llm = FakeLLM(reply="Hello there.")
    pipeline, injector = make_pipeline(config, dictionary, tone_profiles, llm=llm)
    pipeline.start()
    pipeline.submit_chunk(speech_audio())
    import time

    deadline = time.time() + 5
    while not injector.typed and time.time() < deadline:
        time.sleep(0.05)
    pipeline.stop()
    assert injector.typed == ["Hello there. "]  # trailing space appended


def test_process_text_stage2_only(config, dictionary, tone_profiles):
    llm = FakeLLM(reply="Cleaned.")
    pipeline, _ = make_pipeline(config, dictionary, tone_profiles, llm=llm)
    event = pipeline.process_text("uh raw words", app_name="Mail")
    assert event.final_text == "Cleaned."
    assert event.tone == "formal"
    assert event.asr_latency_s == 0.0
