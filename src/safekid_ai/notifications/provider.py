"""Notification Provider Adapters for SafeKid AI Community Group Messaging."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.notifications")


class NotificationProviderType(str, Enum):
    """Supported community notification provider types."""
    NONE = "NONE"
    MOCK = "MOCK"
    WEBHOOK = "WEBHOOK"
    TELEGRAM = "TELEGRAM"


@dataclass
class IncidentAlert:
    """Standardized child-safety incident notification alert payload."""
    incident_id: str
    hazard_zone_name: str
    severity: str
    timestamp_ms: float
    camera_id: str
    track_id: Optional[int] = None
    classification: str = "CHILD"
    confidence: Optional[float] = None
    description: str = ""
    operator_url: Optional[str] = None
    frame_id: Optional[int] = None
    source_video: Optional[str] = None

    def to_formatted_text(self) -> str:
        """Constructs a clean, human-readable alert message for community security groups."""
        sev_emoji = {
            "CRITICAL": "🚨 [CRITICAL HAZARD]",
            "HIGH": "⚠️ [HIGH ALERT]",
            "MEDIUM": "ℹ️ [NOTICE]",
            "LOW": "ℹ️ [INFO]",
        }.get(self.severity.upper(), "⚠️ [ALERT]")

        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp_ms / 1000.0))
        lines = [
            f"{sev_emoji} SafeKid AI Safety Alert",
            f"• Zone: {self.hazard_zone_name}",
            f"• Severity: {self.severity.upper()}",
            f"• Classification: {self.classification} (Track ID: {self.track_id or 'N/A'})",
            f"• Camera / Video: {self.camera_id} ({self.source_video or 'Local Feed'})",
            f"• Time: {time_str}",
            f"• Incident ID: {self.incident_id}",
        ]
        if self.description:
            lines.append(f"• Details: {self.description}")
        if self.operator_url:
            lines.append(f"• Operator Portal: {self.operator_url}")
        return "\n".join(lines)


@dataclass
class NotificationDeliveryResult:
    """Record of a notification dispatch attempt and provider confirmation."""
    success: bool
    provider: str
    message: str
    status_code: Optional[int] = None
    raw_response: Optional[str] = None
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000.0)
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "provider": self.provider,
            "message": self.message,
            "status_code": self.status_code,
            "raw_response": self.raw_response,
            "timestamp_ms": self.timestamp_ms,
            "retry_count": self.retry_count,
        }


class BaseNotificationProvider(ABC):
    """Abstract interface for community notification adapters."""

    @property
    @abstractmethod
    def provider_type(self) -> NotificationProviderType:
        """Unique provider type identifier."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Returns True if the required credentials/endpoints are set."""
        pass

    @abstractmethod
    def send_alert(self, alert: IncidentAlert) -> NotificationDeliveryResult:
        """Dispatches an incident alert to the configured messaging destination."""
        pass

    @abstractmethod
    def send_test(self, message: str = "Test notification from SafeKid AI") -> NotificationDeliveryResult:
        """Dispatches a diagnostic test message."""
        pass


class MockNotificationProvider(BaseNotificationProvider):
    """In-memory mock notification provider for testing and validation without live credentials."""

    def __init__(self, simulate_failure: bool = False):
        self.simulate_failure = simulate_failure
        self.sent_alerts: List[IncidentAlert] = []
        self.sent_messages: List[str] = []

    @property
    def provider_type(self) -> NotificationProviderType:
        return NotificationProviderType.MOCK

    def is_configured(self) -> bool:
        return True

    def send_alert(self, alert: IncidentAlert) -> NotificationDeliveryResult:
        if self.simulate_failure:
            return NotificationDeliveryResult(
                success=False,
                provider="MOCK",
                message="Simulated mock provider dispatch failure",
                status_code=500,
            )
        self.sent_alerts.append(alert)
        return NotificationDeliveryResult(
            success=True,
            provider="MOCK",
            message=f"Mock alert dispatched for incident {alert.incident_id}",
            status_code=200,
            raw_response=json.dumps({"status": "delivered", "mock": True}),
        )

    def send_test(self, message: str = "Test notification from SafeKid AI") -> NotificationDeliveryResult:
        if self.simulate_failure:
            return NotificationDeliveryResult(
                success=False,
                provider="MOCK",
                message="Simulated mock test failure",
                status_code=500,
            )
        self.sent_messages.append(message)
        return NotificationDeliveryResult(
            success=True,
            provider="MOCK",
            message=f"Mock test message dispatched: {message}",
            status_code=200,
            raw_response=json.dumps({"status": "delivered", "mock": True}),
        )


class WebhookNotificationProvider(BaseNotificationProvider):
    """Generic HTTP webhook provider (supports Slack, Discord, Mattermost, or custom community gateways)."""

    def __init__(self, webhook_url: Optional[str] = None, timeout_seconds: float = 5.0):
        self.webhook_url = webhook_url or os.getenv("SAFEKID_WEBHOOK_URL", "").strip()
        self.timeout_seconds = timeout_seconds

    @property
    def provider_type(self) -> NotificationProviderType:
        return NotificationProviderType.WEBHOOK

    def is_configured(self) -> bool:
        return bool(self.webhook_url and self.webhook_url.startswith(("http://", "https://")))

    def _post_payload(self, payload: Dict[str, Any]) -> NotificationDeliveryResult:
        if not self.is_configured():
            return NotificationDeliveryResult(
                success=False,
                provider="WEBHOOK",
                message="Webhook URL is not configured (missing or invalid URL).",
                status_code=None,
            )

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SafeKid-AI-Community-Notifier/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status_code = resp.getcode()
                response_body = resp.read().decode("utf-8", errors="replace")
                success = 200 <= status_code < 300
                return NotificationDeliveryResult(
                    success=success,
                    provider="WEBHOOK",
                    message=f"Webhook returned HTTP {status_code}",
                    status_code=status_code,
                    raw_response=response_body[:500],
                )
        except urllib.error.HTTPError as e:
            return NotificationDeliveryResult(
                success=False,
                provider="WEBHOOK",
                message=f"Webhook HTTP error: {e.code} {e.reason}",
                status_code=e.code,
                raw_response=str(e),
            )
        except Exception as e:
            return NotificationDeliveryResult(
                success=False,
                provider="WEBHOOK",
                message=f"Webhook connection failed: {str(e)}",
                status_code=None,
                raw_response=str(e),
            )

    def send_alert(self, alert: IncidentAlert) -> NotificationDeliveryResult:
        payload = {
            "text": alert.to_formatted_text(),
            "incident_id": alert.incident_id,
            "severity": alert.severity,
            "hazard_zone": alert.hazard_zone_name,
            "classification": alert.classification,
            "camera_id": alert.camera_id,
            "timestamp_ms": alert.timestamp_ms,
        }
        return self._post_payload(payload)

    def send_test(self, message: str = "Test notification from SafeKid AI") -> NotificationDeliveryResult:
        payload = {
            "text": f"🔔 {message} [Diagnostic Test at {time.strftime('%Y-%m-%d %H:%M:%S')}]",
            "type": "diagnostic_test",
        }
        return self._post_payload(payload)


class TelegramNotificationProvider(BaseNotificationProvider):
    """Telegram Bot API provider for gated community central messaging groups."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_seconds: float = 6.0,
    ):
        self.bot_token = bot_token or os.getenv("SAFEKID_TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = chat_id or os.getenv("SAFEKID_TELEGRAM_CHAT_ID", "").strip()
        self.timeout_seconds = timeout_seconds

    @property
    def provider_type(self) -> NotificationProviderType:
        return NotificationProviderType.TELEGRAM

    def is_configured(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _send_telegram_message(self, text: str) -> NotificationDeliveryResult:
        if not self.is_configured():
            return NotificationDeliveryResult(
                success=False,
                provider="TELEGRAM",
                message="Telegram bot token or chat ID is not configured.",
                status_code=None,
            )

        api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SafeKid-AI-Community-Notifier/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status_code = resp.getcode()
                response_body = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(response_body)
                success = parsed.get("ok", False)
                return NotificationDeliveryResult(
                    success=success,
                    provider="TELEGRAM",
                    message="Telegram message delivered" if success else "Telegram API returned ok=false",
                    status_code=status_code,
                    raw_response=response_body[:500],
                )
        except urllib.error.HTTPError as e:
            return NotificationDeliveryResult(
                success=False,
                provider="TELEGRAM",
                message=f"Telegram HTTP error: {e.code} {e.reason}",
                status_code=e.code,
                raw_response=str(e),
            )
        except Exception as e:
            return NotificationDeliveryResult(
                success=False,
                provider="TELEGRAM",
                message=f"Telegram connection failed: {str(e)}",
                status_code=None,
                raw_response=str(e),
            )

    def send_alert(self, alert: IncidentAlert) -> NotificationDeliveryResult:
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(alert.timestamp_ms / 1000.0))
        sev_color = {
            "CRITICAL": "🔴 <b>CRITICAL HAZARD</b>",
            "HIGH": "🟠 <b>HIGH ALERT</b>",
            "MEDIUM": "🟡 <b>NOTICE</b>",
            "LOW": "🔵 <b>INFO</b>",
        }.get(alert.severity.upper(), "⚠️ <b>ALERT</b>")

        html_text = (
            f"{sev_color} — <b>SafeKid AI Community Alert</b>\n\n"
            f"<b>Zone:</b> {alert.hazard_zone_name}\n"
            f"<b>Severity:</b> {alert.severity.upper()}\n"
            f"<b>Classification:</b> {alert.classification} (Track: {alert.track_id or 'N/A'})\n"
            f"<b>Camera/Source:</b> {alert.camera_id} ({alert.source_video or 'Local Feed'})\n"
            f"<b>Time:</b> {time_str}\n"
            f"<b>Incident ID:</b> <code>{alert.incident_id}</code>\n"
        )
        if alert.description:
            html_text += f"<b>Details:</b> {alert.description}\n"
        if alert.operator_url:
            html_text += f"\n<a href='{alert.operator_url}'>Open Operator Control Room</a>"

        return self._send_telegram_message(html_text)

    def send_test(self, message: str = "Test notification from SafeKid AI") -> NotificationDeliveryResult:
        html_text = (
            f"🔔 <b>SafeKid AI Diagnostic Test</b>\n"
            f"Message: {message}\n"
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"<i>Security Control Room Integration Active</i>"
        )
        return self._send_telegram_message(html_text)

