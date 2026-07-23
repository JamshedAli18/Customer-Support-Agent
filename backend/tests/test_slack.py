"""
Standalone test for Slack Incoming Webhooks — verifying both channels
work before wiring them into the real application.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_WARRANTY = os.getenv("SLACK_WEBHOOK_WARRANTY")
SLACK_WEBHOOK_TICKETS = os.getenv("SLACK_WEBHOOK_TICKETS")

if not SLACK_WEBHOOK_WARRANTY:
    raise ValueError("SLACK_WEBHOOK_WARRANTY not found in .env")
if not SLACK_WEBHOOK_TICKETS:
    raise ValueError("SLACK_WEBHOOK_TICKETS not found in .env")


def send_test_message(webhook_url: str, channel_label: str):
    payload = {"text": f"✅ VoiceCart test message — this webhook is correctly posting to {channel_label}."}
    response = requests.post(webhook_url, json=payload)

    if response.status_code == 200 and response.text == "ok":
        print(f"PASS: Message sent successfully to {channel_label}")
    else:
        print(f"FAIL: {channel_label} returned status {response.status_code}: {response.text}")


print("Testing warranty claims webhook...")
send_test_message(SLACK_WEBHOOK_WARRANTY, "#warranty-claims")

print("\nTesting support tickets webhook...")
send_test_message(SLACK_WEBHOOK_TICKETS, "#support-tickets")