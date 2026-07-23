import os
from dotenv import load_dotenv
from cartesia import Cartesia

load_dotenv()

api_key = os.getenv("CARTESIA_API_KEY")
client = Cartesia(api_key=api_key)

VOICE_ID = "694f9389-aac1-45b6-b726-9d9369183238"

questions = [
    ("turn1.wav", "One of my earbuds stopped working completely, this is so annoying"),
    ("turn2.wav", "I already tried that and it's still not charging, I am extremely frustrated"),
]

for filename, text in questions:
    response = client.tts.generate(
        model_id="sonic-3.5",
        transcript=text,
        voice={"mode": "id", "id": VOICE_ID},
        output_format={"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100},
    )
    response.write_to_file(filename)
    print(f"Saved {filename}")