"""Typed internal errors mapped to stable, safe API responses."""


class ModelUnavailableError(RuntimeError):
    """The process is alive but the governed model cannot serve requests."""


class InferenceFailedError(RuntimeError):
    """A validated prediction request failed inside the model service."""


class ExplanationFailedError(RuntimeError):
    """Prediction succeeded or was possible, but explanation construction failed."""


class ExplanationsDisabledError(RuntimeError):
    """The process intentionally disabled the optional explanation endpoint."""
