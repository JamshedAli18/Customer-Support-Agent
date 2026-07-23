import os
from dotenv import load_dotenv
from cartesia import Cartesia

load_dotenv()

api_key = os.getenv("CARTESIA_API_KEY")
client = Cartesia(api_key=api_key)

response = client.tts.generate(
    model_id="sonic-3.5",
    transcript="What is the IPX rating on these earbuds?",
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

response.write_to_file("test_question.wav")
print("Saved test_question.wav")