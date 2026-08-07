"""Process-wide configuration for the local ReturnGuard inference service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_URI = "models:/returnguard-a3@model-of-record"


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


@dataclass(frozen=True)
class Settings:
    """Immutable settings resolved once while the process starts."""

    mlflow_tracking_uri: str
    model_uri: str
    log_level: str
    max_batch_size: int
    explanations_enabled: bool

    @classmethod
    def from_environment(cls) -> "Settings":
        default_tracking_uri = f"sqlite:///{ROOT / '.mlflow' / 'mlflow.db'}"
        max_batch_size = int(os.getenv("RETURNGUARD_MAX_BATCH_SIZE", "100"))
        if not 1 <= max_batch_size <= 100:
            raise ValueError("RETURNGUARD_MAX_BATCH_SIZE must be between 1 and 100")
        return cls(
            mlflow_tracking_uri=os.getenv("RETURNGUARD_MLFLOW_TRACKING_URI", default_tracking_uri),
            model_uri=os.getenv("RETURNGUARD_MODEL_URI", DEFAULT_MODEL_URI),
            log_level=os.getenv("RETURNGUARD_LOG_LEVEL", "INFO").upper(),
            max_batch_size=max_batch_size,
            explanations_enabled=_environment_bool("RETURNGUARD_EXPLANATIONS_ENABLED", True),
        )
