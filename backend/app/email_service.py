"""
Resend email wrapper for VoiceCart support notifications.
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
FROM_ADDRESS = "VoiceCart <onboarding@resend.dev>"


def send_support_email(subject: str, html_body: str) -> str:
    """Sends an email to the support team inbox. Returns the Resend email ID."""
    result = resend.Emails.send({
        "from": FROM_ADDRESS,
        "to": [SUPPORT_EMAIL_TO],
        "subject": subject,
        "html": html_body,
    })
    return result.get("id", "")