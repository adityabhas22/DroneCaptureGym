"""Domain-specific errors used by the environment."""


class DroneCaptureOpsError(Exception):
    """Base error for environment failures."""


class ActionValidationError(DroneCaptureOpsError):
    """Raised when an action does not match the public tool schema."""


class SafetyViolationError(DroneCaptureOpsError):
    """Raised when an action violates safety constraints."""


class EpisodeDoneError(DroneCaptureOpsError):
    """Raised when an action is attempted after an episode has ended."""
