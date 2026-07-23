"""
Slack Incoming Webhook notifications for VoiceCart. Posts formatted
messages to dedicated channels when tickets or warranty claims are
created — separate from (and in addition to) email notifications.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

SLACK_WEBHOOK_WARRANTY = os.getenv("SLACK_WEBHOOK_WARRANTY")
SLACK_WEBHOOK_TICKETS = os.getenv("SLACK_WEBHOOK_TICKETS")
SLACK_WEBHOOK_ORDERS = os.getenv("SLACK_WEBHOOK_ORDERS")


def _post_to_slack(webhook_url: str, text: str) -> bool:
    """Posts a message to a Slack channel via webhook. Returns True on success."""
    if not webhook_url:
        print("[slack_service] No webhook URL configured, skipping notification")
        return False
    try:
        response = requests.post(webhook_url, json={"text": text}, timeout=5)
        return response.status_code == 200 and response.text == "ok"
    except Exception as e:
        print(f"[slack_service] Failed to post to Slack: {e}")
        return False


def notify_support_ticket(ticket_id: str, summary: str, customer_email: str | None) -> bool:
    """Posts a new support ticket notification to #support-tickets."""
    email_line = f"*Customer email:* {customer_email}" if customer_email else "_No email provided_"
    text = (
        f":rotating_light: *New Support Ticket*\n"
        f"*Ticket ID:* `{ticket_id}`\n"
        f"{email_line}\n"
        f"*Summary:* {summary}"
    )
    return _post_to_slack(SLACK_WEBHOOK_TICKETS, text)


def notify_warranty_claim(claim_id: str, earbud_affected: str, issue_description: str, customer_email: str | None) -> bool:
    """Posts a new warranty claim notification to #warranty-claims."""
    email_line = f"*Customer email:* {customer_email}" if customer_email else "_No email provided_"
    text = (
        f":package: *New Warranty Claim*\n"
        f"*Claim ID:* `{claim_id}`\n"
        f"*Earbud affected:* {earbud_affected}\n"
        f"*Issue:* {issue_description}\n"
        f"{email_line}"
    )
    return _post_to_slack(SLACK_WEBHOOK_WARRANTY, text)


def notify_new_order(order_id: str, color: str, quantity: int, total_price: float, customer_email: str) -> bool:
    """Posts a new order notification to #orders."""
    text = (
        f":shopping_trolley: *New Order Booked*\n"
        f"*Order ID:* `{order_id}`\n"
        f"*Product:* ShopNest Pulse — {color} x{quantity}\n"
        f"*Total:* ${total_price:.2f}\n"
        f"*Customer email:* {customer_email}"
    )
    return _post_to_slack(SLACK_WEBHOOK_ORDERS, text)


def notify_order_cancelled(order_id: str, color: str, quantity: int) -> bool:
    """Posts an order cancellation notification to #orders."""
    text = (
        f":x: *Order Cancelled*\n"
        f"*Order ID:* `{order_id}`\n"
        f"*Product:* ShopNest Pulse — {color} x{quantity}\n"
        f"Stock has been restored."
    )
    return _post_to_slack(SLACK_WEBHOOK_ORDERS, text)