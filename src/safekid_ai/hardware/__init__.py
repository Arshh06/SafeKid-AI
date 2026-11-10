"""Hardware adapters and device-control interfaces for SafeKid AI."""

from safekid_ai.hardware.esp32_client import (
    BuzzerTriggerRecord,
    ESP32BuzzerClient,
    ESP32HealthStatus,
)

__all__ = [
    "BuzzerTriggerRecord",
    "ESP32BuzzerClient",
    "ESP32HealthStatus",
]

