"""
Tests app/graph/voice_pipeline.py — validates the full STT -> graph ->
TTS round trip using real audio files. Generates a fresh test audio
file via Cartesia if one isn't already available, so this test is
self-contained and doesn't depend on leftover files from earlier phases.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app", "graph"))
from voice_pipeline import transcribe_audio, synthesize_speech, run_voice_turn

TEST_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio_fixtures")
os.makedirs(TEST_AUDIO_DIR, exist_ok=True)

TEST_INPUT_AUDIO = os.path.join(TEST_AUDIO_DIR, "test_input.wav")
TEST_OUTPUT_AUDIO = os.path.join(TEST_AUDIO_DIR, "test_output.wav")


def ensure_test_audio_exists():
    """Generates a known test audio file via Cartesia if one doesn't already exist."""
    if not os.path.exists(TEST_INPUT_AUDIO):
        print(f"Generating fresh test audio at {TEST_INPUT_AUDIO} ...")
        synthesize_speech(
            "What is the IPX rating on these earbuds?",
            TEST_INPUT_AUDIO,
        )
    return TEST_INPUT_AUDIO


def test_transcribe_audio_produces_text():
    audio_path = ensure_test_audio_exists()
    transcript = transcribe_audio(audio_path)
    assert transcript, "transcribe_audio returned an empty string"
    assert "ipx" in transcript.lower() or "rating" in transcript.lower(), (
        f"Expected transcript to mention IPX/rating, got: '{transcript}'"
    )
    print(f"PASS: transcribe_audio produced a sensible transcript: '{transcript}'")


def test_synthesize_speech_produces_valid_audio_file():
    synthesize_speech("This is a test of the speech synthesis system.", TEST_OUTPUT_AUDIO)
    assert os.path.exists(TEST_OUTPUT_AUDIO), "synthesize_speech did not create an output file"
    size_bytes = os.path.getsize(TEST_OUTPUT_AUDIO)
    assert size_bytes > 1000, f"Output audio file is suspiciously small ({size_bytes} bytes)"
    print(f"PASS: synthesize_speech produced a valid audio file ({size_bytes} bytes)")
    os.remove(TEST_OUTPUT_AUDIO)


def test_full_voice_turn_round_trip():
    """
    Full pipeline: audio in -> transcribe -> graph routing -> grounded
    answer -> synthesize speech out. Validates every piece connects.
    """
    audio_path = ensure_test_audio_exists()
    state = {"messages": [], "sentiment_history": [], "turn_count": 0}

    output_path = os.path.join(TEST_AUDIO_DIR, "round_trip_response.wav")
    result_state = run_voice_turn(audio_path, state, output_audio_path=output_path)

    assert result_state["transcript"], "Expected a non-empty transcript"
    assert result_state["response"], "Expected a non-empty response"
    assert "IPX5" in result_state["response"] or "IPX" in result_state["response"], (
        f"Expected response to mention IPX rating, got: {result_state['response']}"
    )
    assert os.path.exists(output_path), "Expected response audio file to be created"
    assert os.path.getsize(output_path) > 1000, "Response audio file is suspiciously small"

    print(f"PASS: full voice turn round trip works (transcript: '{result_state['transcript']}')")
    os.remove(output_path)


def test_multiturn_voice_state_persists():
    """
    Confirms sentiment_history and turn_count correctly accumulate
    across two separate voice turns using the same state object.
    """
    audio_path = ensure_test_audio_exists()
    state = {"messages": [], "sentiment_history": [], "turn_count": 0}

    output_path_1 = os.path.join(TEST_AUDIO_DIR, "turn1_response.wav")
    output_path_2 = os.path.join(TEST_AUDIO_DIR, "turn2_response.wav")

    state = run_voice_turn(audio_path, state, output_audio_path=output_path_1)
    assert state["turn_count"] == 1, f"Expected turn_count=1 after first turn, got {state['turn_count']}"

    state = run_voice_turn(audio_path, state, output_audio_path=output_path_2)
    assert state["turn_count"] == 2, f"Expected turn_count=2 after second turn, got {state['turn_count']}"
    assert len(state["sentiment_history"]) == 2, "Expected 2 sentiment_history entries after 2 turns"

    print("PASS: multi-turn voice state correctly accumulates across separate voice turns")

    for p in (output_path_1, output_path_2):
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING voice_pipeline.py")
    print("=" * 60)

    test_transcribe_audio_produces_text()
    test_synthesize_speech_produces_valid_audio_file()
    test_full_voice_turn_round_trip()
    test_multiturn_voice_state_persists()

    print("\nAll voice_pipeline.py tests passed.")