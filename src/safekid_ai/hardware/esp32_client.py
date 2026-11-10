"""ESP32 Physical Buzzer Control Client for SafeKid AI."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

from safekid_ai.utils.logger import get_logger

logger = get_logger("safekid_ai.hardware.esp32")


@dataclass
class ESP32HealthStatus:
    """Connection status and diagnostics for the ESP32 physical buzzer device."""
    connected: bool
    status_label: str  # "CONNECTED", "NOT CONNECTED", "OFFLINE", "UNCONFIGURED"
    host: str
    port: int
    latency_ms: Optional[float] = None
    firmware_version: Optional[str] = None
    gpio_pin: Optional[int] = None
    last_checked_ms: float = field(default_factory=lambda: time.time() * 1000.0)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "status": self.status_label,
            "host": self.host,
            "port": self.port,
            "latency_ms": self.latency_ms,
            "firmware_version": self.firmware_version,
            "gpio_pin": self.gpio_pin,
            "last_checked_ms": self.last_checked_ms,
            "error_message": self.error_message,
        }


@dataclass
class BuzzerTriggerRecord:
    """Audit record of a physical buzzer actuation attempt."""
    timestamp_ms: float
    command: str  # "TRIGGER", "SILENCE", "TEST"
    duration_ms: int
    incident_id: Optional[str]
    severity: Optional[str]
    success: bool
    http_status: Optional[int]
    device_response: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp_ms / 1000.0)),
            "timestamp_ms": self.timestamp_ms,
            "command": self.command,
            "duration_ms": self.duration_ms,
            "incident_id": self.incident_id,
            "severity": self.severity,
            "success": self.success,
            "http_status": self.http_status,
            "device_response": self.device_response,
            "error": self.error,
        }


class ESP32BuzzerClient:
    """Client for controlling an ESP32 microcontroller with a physical buzzer over HTTP REST.

    Endpoints on ESP32 firmware:
      - GET  /health          -> Returns {"status": "ok", "device": "SafeKid-ESP32-Buzzer", "gpio": 18, "fw": "1.0"}
      - POST /buzzer/trigger  -> Payload: {"duration_ms": 1500, "reason": "...", "severity": "CRITICAL"}
      - POST /buzzer/silence  -> Immediate GPIO LOW silence
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 80,
        auth_token: Optional[str] = None,
        timeout_seconds: float = 2.0,
        cooldown_seconds: float = 10.0,
        config_path: Optional[Path] = None,
        history_path: Optional[Path] = None,
    ):
        self.config_path = config_path or Path("data/config/esp32_config.json")
        self.history_path = history_path or Path("data/hardware/buzzer_history.json")

        self.host = host or os.getenv("SAFEKID_ESP32_HOST", "192.168.1.120").strip()
        self.port = int(os.getenv("SAFEKID_ESP32_PORT", str(port)))
        self.auth_token = auth_token or os.getenv("SAFEKID_ESP32_TOKEN", "").strip()
        self.timeout_seconds = timeout_seconds
        self.cooldown_seconds = cooldown_seconds
        self.default_duration_ms = 1500

        self._last_trigger_ms: float = 0.0
        self._history: List[BuzzerTriggerRecord] = []
        self._last_health: Optional[ESP32HealthStatus] = None

        self._ensure_dirs()
        self.load_config()
        self._load_history()

    def _ensure_dirs(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def load_config(self) -> None:
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.host = data.get("host", self.host)
                self.port = int(data.get("port", self.port))
                self.auth_token = data.get("auth_token", self.auth_token)
                self.cooldown_seconds = float(data.get("cooldown_seconds", self.cooldown_seconds))
                self.default_duration_ms = int(data.get("default_duration_ms", self.default_duration_ms))
                logger.info(f"Loaded ESP32 config: host={self.host}, port={self.port}")
            except Exception as e:
                logger.warning(f"Could not load ESP32 config from {self.config_path}: {e}")

    def save_config(
        self,
        host: str,
        port: int = 80,
        auth_token: Optional[str] = None,
        cooldown_seconds: float = 10.0,
        default_duration_ms: int = 1500,
    ) -> None:
        self.host = host.strip()
        self.port = int(port)
        if auth_token is not None:
            self.auth_token = auth_token.strip()
        self.cooldown_seconds = float(cooldown_seconds)
        self.default_duration_ms = int(default_duration_ms)

        cfg_data = {
            "host": self.host,
            "port": self.port,
            "auth_token": self.auth_token,
            "cooldown_seconds": self.cooldown_seconds,
            "default_duration_ms": self.default_duration_ms,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)
        logger.info(f"Saved ESP32 configuration to {self.config_path}")

    def _load_history(self) -> None:
        if self.history_path.exists():
            try:
                with open(self.history_path, "r", encoding="utf-8") as f:
                    raw_list = json.load(f)
                for item in raw_list:
                    self._history.append(
                        BuzzerTriggerRecord(
                            timestamp_ms=item.get("timestamp_ms", 0.0),
                            command=item.get("command", "TRIGGER"),
                            duration_ms=item.get("duration_ms", 1500),
                            incident_id=item.get("incident_id"),
                            severity=item.get("severity"),
                            success=item.get("success", False),
                            http_status=item.get("http_status"),
                            device_response=item.get("device_response", ""),
                            error=item.get("error"),
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not load buzzer history: {e}")

    def _record_action(self, record: BuzzerTriggerRecord) -> None:
        self._history.insert(0, record)
        if len(self._history) > 100:
            self._history = self._history[:100]
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self._history], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist buzzer history: {e}")

    def check_health(self) -> ESP32HealthStatus:
        """Pings the ESP32 /health endpoint to verify physical hardware responsiveness."""
        if not self.host:
            status = ESP32HealthStatus(
                connected=False,
                status_label="UNCONFIGURED",
                host="",
                port=self.port,
                error_message="ESP32 host is empty.",
            )
            self._last_health = status
            return status

        url = f"http://{self.host}:{self.port}/health"
        req = urllib.request.Request(url, headers={"User-Agent": "SafeKid-AI-ESP32-Client/1.0"})
        if self.auth_token:
            req.add_header("Authorization", f"Bearer {self.auth_token}")

        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                elapsed_ms = (time.time() - t0) * 1000.0
                body = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(body) if body.startswith("{") else {}
                status = ESP32HealthStatus(
                    connected=True,
                    status_label="CONNECTED",
                    host=self.host,
                    port=self.port,
                    latency_ms=round(elapsed_ms, 1),
                    firmware_version=parsed.get("fw", "1.0"),
                    gpio_pin=parsed.get("gpio", 18),
                )
                self._last_health = status
                return status
        except urllib.error.HTTPError as e:
            status = ESP32HealthStatus(
                connected=False,
                status_label="NOT CONNECTED",
                host=self.host,
                port=self.port,
                error_message=f"HTTP {e.code}: {e.reason}",
            )
            self._last_health = status
            return status
        except Exception as e:
            status = ESP32HealthStatus(
                connected=False,
                status_label="NOT CONNECTED",
                host=self.host,
                port=self.port,
                error_message=f"Connection failed: {str(e)}",
            )
            self._last_health = status
            return status

    def trigger_buzzer(
        self,
        duration_ms: Optional[int] = None,
        incident_id: Optional[str] = None,
        severity: str = "CRITICAL",
        reason: str = "Child entered hazardous zone",
        command_type: str = "TRIGGER",
    ) -> BuzzerTriggerRecord:
        """Sends an activation signal to the ESP32 physical buzzer."""
        duration = duration_ms or self.default_duration_ms
        now_ms = time.time() * 1000.0

        # Check cooldown (only apply cooldown to automated triggers, not manual test triggers)
        if command_type == "TRIGGER" and (now_ms - self._last_trigger_ms) < (self.cooldown_seconds * 1000.0):
            record = BuzzerTriggerRecord(
                timestamp_ms=now_ms,
                command=command_type,
                duration_ms=duration,
                incident_id=incident_id,
                severity=severity,
                success=False,
                http_status=None,
                device_response="Suppressed: cooldown window active",
                error=f"Cooldown active ({self.cooldown_seconds}s interval).",
            )
            self._record_action(record)
            return record

        if not self.host:
            record = BuzzerTriggerRecord(
                timestamp_ms=now_ms,
                command=command_type,
                duration_ms=duration,
                incident_id=incident_id,
                severity=severity,
                success=False,
                http_status=None,
                device_response="Failed: ESP32 host unconfigured",
                error="ESP32 host IP is not configured.",
            )
            self._record_action(record)
            return record

        url = f"http://{self.host}:{self.port}/buzzer/trigger"
        payload = {
            "duration_ms": duration,
            "incident_id": incident_id or "",
            "severity": severity,
            "reason": reason,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SafeKid-AI-ESP32-Client/1.0",
            },
            method="POST",
        )
        if self.auth_token:
            req.add_header("Authorization", f"Bearer {self.auth_token}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                status_code = resp.getcode()
                success = 200 <= status_code < 300
                if success:
                    self._last_trigger_ms = now_ms

                record = BuzzerTriggerRecord(
                    timestamp_ms=now_ms,
                    command=command_type,
                    duration_ms=duration,
                    incident_id=incident_id,
                    severity=severity,
                    success=success,
                    http_status=status_code,
                    device_response=body[:200],
                )
                self._record_action(record)
                return record
        except urllib.error.HTTPError as e:
            record = BuzzerTriggerRecord(
                timestamp_ms=now_ms,
                command=command_type,
                duration_ms=duration,
                incident_id=incident_id,
                severity=severity,
                success=False,
                http_status=e.code,
                device_response="",
                error=f"HTTP {e.code}: {e.reason}",
            )
            self._record_action(record)
            return record
        except Exception as e:
            record = BuzzerTriggerRecord(
                timestamp_ms=now_ms,
                command=command_type,
                duration_ms=duration,
                incident_id=incident_id,
                severity=severity,
                success=False,
                http_status=None,
                device_response="",
                error=f"Hardware connection failed: {str(e)}",
            )
            self._record_action(record)
            return record

    def silence_buzzer(self) -> BuzzerTriggerRecord:
        """Sends an immediate silence signal to cut the buzzer GPIO low."""
        now_ms = time.time() * 1000.0
        if not self.host:
            record = BuzzerTriggerRecord(
                timestamp_ms=now_ms,
                command="SILENCE",
                duration_ms=0,
                incident_id=None,
                severity=None,
                success=False,
                http_status=None,
                device_response="Failed: host unconfigured",
                error="ESP32 host IP is not configured.",
            )
            self._record_action(record)
            return record

        url = f"http://{self.host}:{self.port}/buzzer/silence"
        req = urllib.request.Request(
            url,
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SafeKid-AI-ESP32-Client/1.0",
            },
            method="POST",
        )
        if self.auth_token:
            req.add_header("Authorization", f"Bearer {self.auth_token}")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                status_code = resp.getcode()
                record = BuzzerTriggerRecord(
                    timestamp_ms=now_ms,
                    command="SILENCE",
                    duration_ms=0,
                    incident_id=None,
                    severity=None,
                    success=200 <= status_code < 300,
                    http_status=status_code,
                    device_response=body[:200],
                )
                self._record_action(record)
                return record
        except Exception as e:
            record = BuzzerTriggerRecord(
                timestamp_ms=now_ms,
                command="SILENCE",
                duration_ms=0,
                incident_id=None,
                severity=None,
                success=False,
                http_status=None,
                device_response="",
                error=f"Silence command failed: {str(e)}",
            )
            self._record_action(record)
            return record

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state and configuration of the buzzer client."""
        # Check health if not checked recently (< 10s)
        if not self._last_health or (time.time() * 1000.0 - self._last_health.last_checked_ms) > 10000.0:
            self.check_health()

        token_masked = ""
        if self.auth_token:
            token_masked = self.auth_token[:3] + "..." + self.auth_token[-2:] if len(self.auth_token) > 5 else "******"

        return {
            "health": self._last_health.to_dict() if self._last_health else {"connected": False, "status": "NOT CONNECTED"},
            "config": {
                "host": self.host,
                "port": self.port,
                "token_masked": token_masked,
                "cooldown_seconds": self.cooldown_seconds,
                "default_duration_ms": self.default_duration_ms,
            },
            "recent_history": [r.to_dict() for r in self._history[:15]],
        }

