"""
Run this ONCE to authorize Gmail sending. Opens a browser for you to log
into your Gmail account and grant permission. Produces token.json, which
is then reused automatically (and refreshed) by email_service.py — no
need to run this script again unless token.json is deleted or revoked.
"""

import os
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "token.json")

flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
creds = flow.run_local_server(port=0)

with open(TOKEN_PATH, "w") as token_file:
    token_file.write(creds.to_json())

print(f"Authorization complete. Token saved to {TOKEN_PATH}")