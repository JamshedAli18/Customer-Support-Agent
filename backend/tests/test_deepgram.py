"""
Standalone test for Deepgram Aura TTS, explicitly requesting WAV output
(linear16 encoding) so it matches Cartesia's WAV format — keeps the
rest of the pipeline (frontend audio playback) format-agnostic.
"""

import os
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

api_key = os.getenv("DEEPGRAM_API_KEY")
if not api_key:
    raise ValueError("DEEPGRAM_API_KEY not found in .env")

client = DeepgramClient(api_key=api_key)

response = client.speak.v1.audio.generate(
    text="Hello Jamshed, this is a test of Deepgram text to speech as a fallback for VoiceCart.",
    model="aura-2-asteria-en",
    encoding="linear16",
    container="wav",
    sample_rate=44100,
)

audio_bytes = b"".join(response)

output_path = os.path.join(os.path.dirname(__file__), "deepgram_test_output_wav.wav")
with open(output_path, "wb") as f:
    f.write(audio_bytes)

size_kb = len(audio_bytes) / 1024
print(f"PASS: Deepgram TTS generated WAV audio successfully ({size_kb:.1f} KB)")
print(f"Saved to: {output_path}")

# Sanity check: confirm it actually starts with a RIFF/WAV header
if audio_bytes[:4] == b"RIFF":
    print("PASS: Output has a valid RIFF/WAV header")
else:
    print(f"WARNING: Expected RIFF header, got: {audio_bytes[:4]}")