"""
Tests the Cartesia -> Deepgram TTS fallback by forcing Cartesia to fail.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "app", "graph"))

# Temporarily break Cartesia by pointing it at a bad model_id, forcing
# the fallback path. We monkeypatch the internal function for this test.
import voice_pipeline as vp


def test_fallback_triggers_on_cartesia_failure():
    original = vp._synthesize_with_cartesia

    def broken_cartesia(*args, **kwargs):
        raise RuntimeError("Simulated Cartesia failure (e.g. rate limit)")

    vp._synthesize_with_cartesia = broken_cartesia

    try:
        output_path = os.path.join(os.path.dirname(__file__), "fallback_test_output.wav")
        result_path = vp.synthesize_speech("This should come from Deepgram instead.", output_path)

        assert os.path.exists(result_path), "Expected output file to be created via fallback"
        with open(result_path, "rb") as f:
            header = f.read(4)
        assert header == b"RIFF", f"Expected valid WAV (RIFF) header, got {header}"

        print(f"PASS: Cartesia failure correctly triggered Deepgram fallback ({os.path.getsize(result_path)} bytes)")
        os.remove(result_path)
    finally:
        vp._synthesize_with_cartesia = original


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING Cartesia -> Deepgram TTS fallback")
    print("=" * 60)
    test_fallback_triggers_on_cartesia_failure()
    print("\nFallback test passed.")