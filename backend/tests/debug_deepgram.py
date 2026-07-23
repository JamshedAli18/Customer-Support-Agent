import os
from dotenv import load_dotenv
from deepgram import DeepgramClient

load_dotenv()

api_key = os.getenv("DEEPGRAM_API_KEY")
client = DeepgramClient(api_key=api_key)

response = client.speak.v1.audio.generate(
    text="Hello, this is a test.",
    model="aura-2-asteria-en",
)

print(f"Type of response: {type(response)}")

chunks = list(response)
print(f"Number of chunks: {len(chunks)}")
if chunks:
    print(f"Type of first chunk: {type(chunks[0])}")
    print(f"First chunk preview: {chunks[0][:50] if isinstance(chunks[0], bytes) else chunks[0]}")