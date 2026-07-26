"""
voice_pipeline.py — wraps the text-based LangGraph agent with audio
input/output. STT (Groq Whisper) -> graph.invoke() -> TTS (Cartesia).
"""

import os
from dotenv import load_dotenv
from groq import Groq
from cartesia import Cartesia

import sys
sys.path.append(os.path.dirname(__file__))
from graph import graph

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
cartesia_client = Cartesia(api_key=CARTESIA_API_KEY, timeout=8.0)

DEFAULT_VOICE_ID = "694f9389-aac1-45b6-b726-9d9369183238"


def transcribe_audio(audio_file_path: str) -> str:
    """Transcribes an audio file to text using Groq Whisper."""
    with open(audio_file_path, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(os.path.basename(audio_file_path), file.read()),
            model="whisper-large-v3",
            prompt="The following is a customer support inquiry from a user. Please transcribe it accurately. Ignore any background noise or silence.",
            response_format="text",
            language="en",
            temperature=0.0,
        )
    return transcription.strip()


import requests
from deepgram import DeepgramClient

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
deepgram_client = DeepgramClient(api_key=DEEPGRAM_API_KEY) if DEEPGRAM_API_KEY else None

DEEPGRAM_VOICE_MODEL = "aura-2-asteria-en"


def _synthesize_with_cartesia(text: str, output_path: str, voice_id: str = DEFAULT_VOICE_ID) -> str:
    response = cartesia_client.tts.generate(
        model_id="sonic-turbo",
        transcript=text,
        voice={"mode": "id", "id": voice_id},
        output_format={
            "container": "wav",
            "encoding": "pcm_f32le",
            "sample_rate": 44100,
        },
    )
    response.write_to_file(output_path)
    return output_path


def _synthesize_with_deepgram(text: str, output_path: str) -> str:
    response = requests.post(
        f"https://api.deepgram.com/v1/speak?model={DEEPGRAM_VOICE_MODEL}&encoding=linear16&container=wav&sample_rate=44100",
        headers={
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"text": text},
        timeout=(3, 6),
    )
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path


def synthesize_speech(text: str, output_path: str, voice_id: str = DEFAULT_VOICE_ID) -> str:
    """
    Converts text to speech, preferring Cartesia. Falls back to Deepgram
    Aura on any Cartesia failure (rate limit, timeout, outage, etc.) so
    the pipeline stays resilient. Both providers return WAV, so the
    output format is consistent regardless of which one was used.
    """
    try:
        return _synthesize_with_cartesia(text, output_path, voice_id)
    except Exception as e:
        print(f"[voice_pipeline] Cartesia TTS failed ({e}), falling back to Deepgram")
        if deepgram_client is None:
            raise RuntimeError(
                "Cartesia failed and DEEPGRAM_API_KEY is not configured for fallback"
            ) from e
        return _synthesize_with_deepgram(text, output_path)


def run_voice_turn(audio_file_path: str, state: dict, output_audio_path: str = "response.wav") -> dict:
    """
    Full voice turn: transcribe audio -> run through graph -> synthesize response.
    Mutates and returns the conversation state, plus adds 'audio_output_path'.
    """
    transcript = transcribe_audio(audio_file_path)
    print(f"[voice_pipeline] Transcribed: {transcript}")

    state["messages"].append({"role": "user", "content": transcript})
    state = graph.invoke(state)

    response_text = state.get("response", "I'm sorry, I didn't catch that.")
    audio_path = synthesize_speech(response_text, output_audio_path)
    print(f"[voice_pipeline] Response audio saved: {audio_path}")

    state["messages"].append({"role": "assistant", "content": response_text})
    state["audio_output_path"] = audio_path
    state["transcript"] = transcript

    return state


if __name__ == "__main__":
    # Reuse the test_question.wav we already generated and verified
    test_state = {"messages": [], "sentiment_history": [], "turn_count": 0}

    result = run_voice_turn("test_question.wav", test_state, output_audio_path="test_response.wav")

    print(f"\nTranscript: {result['transcript']}")
    print(f"Response text: {result['response']}")
    print(f"Response audio: {result['audio_output_path']}")