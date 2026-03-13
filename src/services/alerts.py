"""
Retail Brain — Alerting Service
Dispatches critical stockout notifications via Email, Slack Webhooks, or SMS.
"""

import os
import smtplib
from email.message import EmailMessage
import requests
from typing import List, Dict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from logger import get_logger

logger = get_logger("services.alerts")


class AlertDispatcher:
    """Handles routing alerts to configured enterprise channels."""
    
    def __init__(self):
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_password = os.getenv("SMTP_PASSWORD")

    def format_alert_message(self, critical_products: List[Dict], store_id: str) -> str:
        """Formats the list of top at-risk products into a readable message."""
        msg = f"🚨 *CRITICAL STOCKOUT ALERT* for Store: {store_id} 🚨\n\n"
        for p in critical_products:
            msg += f"• *{p['product_name']}*\n"
            msg += f"  - Stock on hand: {p['stock_on_hand']}\n"
            msg += f"  - Reorder point: {p.get('reorder_point', 'N/A')}\n"
            msg += f"  - Probability: {p['stockout_probability']*100:.1f}%\n"
            msg += f"  - Days of cover: {p['days_of_cover']}d\n\n"
        
        msg += "Please order replenishments immediately."
        return msg

    def send_slack_alert(self, message: str) -> bool:
        """Sends a formatted message to a Slack channel via webhook."""
        if not self.slack_webhook_url:
            logger.warning("[SLACK ALERT (Simulated)] webhook URL not configured.")
            logger.info(f"Message that would have been sent:\n{message}")
            return True
            
        try:
            response = requests.post(
                self.slack_webhook_url,
                json={"text": message},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            logger.info("Slack alert sent successfully.")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False

    def send_email_alert(self, recipient_email: str, subject: str, message: str) -> bool:
        """Sends an email alert using SMTP."""
        if not all([self.smtp_user, self.smtp_password]):
            logger.warning(f"[EMAIL ALERT (Simulated)] to {recipient_email}")
            logger.info(f"Subject: {subject}\nMessage:\n{message}")
            return True

        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = subject
        msg['From'] = self.smtp_user
        msg['To'] = recipient_email

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            logger.info(f"Email alert sent successfully to {recipient_email}.")
            return True
        except Exception as e:
            logger.error(f"Failed to send Email alert: {e}")
            return False

    def dispatch_critical_stockouts(self, critical_products: List[Dict], store_id: str, manager_email: str = None):
        """Top-level function called by the APScheduler when critical risks are found."""
        if not critical_products:
            logger.info(f"No critical stockouts for {store_id}. No alerts sent.")
            return
            
        logger.info(f"Dispatching alerts for {len(critical_products)} critical items at {store_id}.")
        message = self.format_alert_message(critical_products, store_id)
        
        # 1. Send Slack
        self.send_slack_alert(message)
        
        # 2. Send Email
        if manager_email:
            self.send_email_alert(manager_email, f"URGENT: Stockout Risks - {store_id}", message)


# Singleton instance for easy importing
dispatcher = AlertDispatcher()
