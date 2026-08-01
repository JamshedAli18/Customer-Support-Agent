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


import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

GMAIL_TOKEN_PATH = "/etc/secrets/token.json" if os.path.exists("/etc/secrets/token.json") else os.path.join(os.path.dirname(__file__), "token.json")
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


def _get_gmail_service():
    creds = Credentials.from_authorized_user_file(GMAIL_TOKEN_PATH, GMAIL_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(GMAIL_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def send_customer_email(to_email: str, subject: str, html_body: str) -> bool:
    """Sends an email to a real customer address via the Gmail API."""
    try:
        service = _get_gmail_service()
        message = MIMEText(html_body, "html")
        message["to"] = to_email
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        return True
    except Exception as e:
        print(f"[send_customer_email] Gmail send failed: {e}")
        return False