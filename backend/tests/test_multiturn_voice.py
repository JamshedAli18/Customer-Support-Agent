import os
import sys

# Add parent directory to path to allow importing app.graph
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.graph.voice_pipeline import run_voice_turn
    
if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-TURN VOICE ESCALATION TEST")
    print("=" * 60)

    state = {"messages": [], "sentiment_history": [], "turn_count": 0}

    print("\n--- Turn 1 (audio: turn1.wav) ---")
    state = run_voice_turn("turn1.wav", state, output_audio_path="turn1_response.wav")
    print(f"Transcript: {state['transcript']}")
    print(f"Response: {state['response']}")
    print(f"Sentiment history: {state.get('sentiment_history')}")
    print(f"Escalated: {state.get('escalated', False)}")

    print("\n--- Turn 2 (audio: turn2.wav) ---")
    state = run_voice_turn("turn2.wav", state, output_audio_path="turn2_response.wav")
    print(f"Transcript: {state['transcript']}")
    print(f"Response: {state['response']}")
    print(f"Sentiment history: {state.get('sentiment_history')}")
    print(f"Escalated: {state.get('escalated', False)}")
    if state.get("escalated"):
        print(f"Ticket ID: {state.get('ticket_id')}")