import os
from dotenv import load_dotenv
from cartesia import Cartesia

load_dotenv()

api_key = os.getenv("CARTESIA_API_KEY")
if not api_key:
    raise ValueError("CARTESIA_API_KEY not found in .env")

client = Cartesia(api_key=api_key)

response = client.tts.generate(
    model_id="sonic-3.5",
    transcript="Hello Jamshed, this is a test of Cartesia text to speech for your VoiceCart project.",
    voice={
        "mode": "id",
        "id": "694f9389-aac1-45b6-b726-9d9369183238",
    },
    output_format={
        "container": "wav",
        "encoding": "pcm_f32le",
        "sample_rate": 44100,
    },
)

response.write_to_file("cartesia_test_output.wav")
print("Success — audio saved as cartesia_test_output.wav")