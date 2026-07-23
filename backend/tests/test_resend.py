"""
Standalone test for Resend email sending — verifying it works before
wiring it into the real application.
"""

import os
from dotenv import load_dotenv
import resend

load_dotenv()

RESEND_API_KEY = os.getenv("RESEND_API_KEY")
SUPPORT_EMAIL_TO = os.getenv("SUPPORT_EMAIL_TO")

if not RESEND_API_KEY:
    raise ValueError("RESEND_API_KEY not found in .env")
if not SUPPORT_EMAIL_TO:
    raise ValueError("SUPPORT_EMAIL_TO not found in .env")

resend.api_key = RESEND_API_KEY

params = {
    "from": "VoiceCart <onboarding@resend.dev>",
    "to": [SUPPORT_EMAIL_TO],
    "subject": "VoiceCart — Resend test email",
    "html": """
        <h2>VoiceCart Email Test</h2>
        <p>This is a test email from the VoiceCart project setup.</p>
        <p>If you're reading this, <strong>Resend is correctly configured</strong>.</p>
    """,
}

email = resend.Emails.send(params)
print(f"PASS: Email sent successfully")
print(f"Email ID: {email['id']}")
print(f"Sent to: {SUPPORT_EMAIL_TO}")
print(f"\nCheck your inbox (and spam folder) for the test email.")