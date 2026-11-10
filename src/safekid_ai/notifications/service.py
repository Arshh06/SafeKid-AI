"""Notification Orchestration Service for SafeKid AI."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

from safekid_ai.notifications.provider import (
    BaseNotificationProvider,
    IncidentAlert,
    MockNotificationProvider,
    NotificationDeliveryResult,
    NotificationProviderType,
    TelegramNotificationProvider,
    WebhookNotificationProvider,
)
from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.notifications.service")


class NotificationService:
    """Manages community notification dispatch, delivery logs, rate-limiting, and deduplication."""

    def __init__(
        self,
        config_path: Optional[Path] = None,
        audit_log_path: Optional[Path] = None,
        default_provider: NotificationProviderType = NotificationProviderType.NONE,
    ):
        self.config_path = config_path or Path("data/config/notifications.json")
        self.audit_log_path = audit_log_path or Path("data/notifications/audit_log.json")
        self._provider: BaseNotificationProvider = self._create_provider(default_provider)
        self._recent_dispatches: Dict[Tuple[str, str], float] = {}  # (incident_id, zone_name) -> timestamp_ms
        self._history: List[Dict[str, Any]] = []

        self._ensure_dirs()
        self.load_config()
        self._load_audit_log()

    def _ensure_dirs(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_provider(
        self,
        provider_type: NotificationProviderType,
        webhook_url: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> BaseNotificationProvider:
        if provider_type == NotificationProviderType.WEBHOOK:
            return WebhookNotificationProvider(webhook_url=webhook_url)
        elif provider_type == NotificationProviderType.TELEGRAM:
            return TelegramNotificationProvider(bot_token=telegram_token, chat_id=telegram_chat_id)
        elif provider_type == NotificationProviderType.MOCK:
            return MockNotificationProvider()
        else:
            # NONE provider: unconfigured dummy provider
            class NoneNotificationProvider(BaseNotificationProvider):
                @property
                def provider_type(self) -> NotificationProviderType:
                    return NotificationProviderType.NONE

                def is_configured(self) -> bool:
                    return False

                def send_alert(self, alert: IncidentAlert) -> NotificationDeliveryResult:
                    return NotificationDeliveryResult(
                        success=False,
                        provider="NONE",
                        message="Notification service is NOT CONFIGURED.",
                    )

                def send_test(self, message: str = "") -> NotificationDeliveryResult:
                    return NotificationDeliveryResult(
                        success=False,
                        provider="NONE",
                        message="Notification service is NOT CONFIGURED.",
                    )

            return NoneNotificationProvider()

    def load_config(self) -> None:
        """Loads provider configuration from environment or config file."""
        # 1. Environment variables take precedence
        env_webhook = os.getenv("SAFEKID_WEBHOOK_URL", "").strip()
        env_tg_token = os.getenv("SAFEKID_TELEGRAM_BOT_TOKEN", "").strip()
        env_tg_chat = os.getenv("SAFEKID_TELEGRAM_CHAT_ID", "").strip()

        if env_webhook:
            self._provider = WebhookNotificationProvider(webhook_url=env_webhook)
            logger.info("Notification provider initialized as WEBHOOK from environment variables.")
            return
        elif env_tg_token and env_tg_chat:
            self._provider = TelegramNotificationProvider(bot_token=env_tg_token, chat_id=env_tg_chat)
            logger.info("Notification provider initialized as TELEGRAM from environment variables.")
            return

        # 2. Check JSON config file
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ptype = NotificationProviderType(data.get("provider_type", "NONE"))
                self._provider = self._create_provider(
                    provider_type=ptype,
                    webhook_url=data.get("webhook_url"),
                    telegram_token=data.get("telegram_token"),
                    telegram_chat_id=data.get("telegram_chat_id"),
                )
                logger.info(f"Loaded notification config from {self.config_path}: {ptype.value}")
            except Exception as e:
                logger.warning(f"Failed to load notification config from {self.config_path}: {e}")

    def save_config(
        self,
        provider_type: str,
        webhook_url: Optional[str] = None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> None:
        """Saves provider configuration to file."""
        ptype = NotificationProviderType(provider_type.upper())
        self._provider = self._create_provider(
            provider_type=ptype,
            webhook_url=webhook_url,
            telegram_token=telegram_token,
            telegram_chat_id=telegram_chat_id,
        )

        cfg_data = {
            "provider_type": ptype.value,
            "webhook_url": webhook_url or "",
            "telegram_token": telegram_token or "",
            "telegram_chat_id": telegram_chat_id or "",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)
        logger.info(f"Saved notification configuration to {self.config_path}")

    def _load_audit_log(self) -> None:
        if self.audit_log_path.exists():
            try:
                with open(self.audit_log_path, "r", encoding="utf-8") as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load notification audit log: {e}")
                self._history = []

    def _append_audit_log(self, record: Dict[str, Any]) -> None:
        self._history.insert(0, record)
        if len(self._history) > 200:
            self._history = self._history[:200]
        try:
            with open(self.audit_log_path, "w", encoding="utf-8") as f:
                json.dump(self._history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to write notification audit log: {e}")

    def dispatch_incident_alert(
        self, alert: IncidentAlert, min_interval_seconds: float = 60.0
    ) -> NotificationDeliveryResult:
        """Dispatches an alert with deduplication safeguards."""
        key = (alert.incident_id, alert.hazard_zone_name)
        now_ms = time.time() * 1000.0
        last_ms = self._recent_dispatches.get(key, 0.0)

        # Check deduplication window
        if (now_ms - last_ms) < (min_interval_seconds * 1000.0):
            res = NotificationDeliveryResult(
                success=False,
                provider=self._provider.provider_type.value,
                message=f"Suppressed duplicate alert within {min_interval_seconds}s cooldown window.",
                status_code=None,
            )
            return res

        # Attempt dispatch via active provider
        result = self._provider.send_alert(alert)
        if result.success:
            self._recent_dispatches[key] = now_ms

        # Record audit log entry
        log_entry = {
            "type": "INCIDENT_ALERT",
            "incident_id": alert.incident_id,
            "zone": alert.hazard_zone_name,
            "severity": alert.severity,
            "classification": alert.classification,
            "success": result.success,
            "provider": result.provider,
            "status_code": result.status_code,
            "message": result.message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._append_audit_log(log_entry)
        return result

    def send_test_message(self, message: str = "Test notification from SafeKid AI") -> NotificationDeliveryResult:
        """Sends a diagnostic test notification."""
        result = self._provider.send_test(message)
        log_entry = {
            "type": "TEST_MESSAGE",
            "message_text": message,
            "success": result.success,
            "provider": result.provider,
            "status_code": result.status_code,
            "message": result.message,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._append_audit_log(log_entry)
        return result

    def get_status(self) -> Dict[str, Any]:
        """Returns the current operational status of the notification adapter."""
        ptype = self._provider.provider_type.value
        configured = self._provider.is_configured()

        total_sent = sum(1 for h in self._history if h.get("success"))
        total_failed = sum(1 for h in self._history if not h.get("success"))

        status_str = "CONFIGURED" if configured else "NOT CONFIGURED"
        if ptype == "NONE":
            status_str = "NOT CONFIGURED"

        # Mask credentials for display
        webhook_masked = ""
        telegram_token_masked = ""
        telegram_chat_masked = ""

        if isinstance(self._provider, WebhookNotificationProvider) and self._provider.webhook_url:
            u = self._provider.webhook_url
            webhook_masked = u[:15] + "..." + u[-8:] if len(u) > 25 else "********"
        elif isinstance(self._provider, TelegramNotificationProvider):
            t = self._provider.bot_token
            c = self._provider.chat_id
            telegram_token_masked = t[:6] + "..." + t[-4:] if len(t) > 12 else "********"
            telegram_chat_masked = c[:3] + "..." + c[-2:] if len(c) > 6 else "********"

        return {
            "status": status_str,
            "provider_type": ptype,
            "is_configured": configured,
            "webhook_url_masked": webhook_masked,
            "telegram_token_masked": telegram_token_masked,
            "telegram_chat_id_masked": telegram_chat_masked,
            "total_sent": total_sent,
            "total_failed": total_failed,
            "last_event": self._history[0] if self._history else None,
        }

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._history[:limit]

