"""Drift-response policy for closed-loop adaptation."""

import enum


class Action(str, enum.Enum):
    """The response the policy selects for an observed drift band."""

    NONE = "none"
    ALERT = "alert"
    RETRAIN = "retrain"


def decide(band: str) -> Action:
    """Map a PSI drift band to the response action."""
    if band == "significant":
        return Action.RETRAIN
    if band == "moderate":
        return Action.ALERT
    return Action.NONE


def in_cooldown(
    last_retrain_ts: float | None, now: float, cooldown_seconds: float
) -> bool:
    """Return True if a retrain happened too recently to trigger another."""
    if last_retrain_ts is None:
        return False
    return (now - last_retrain_ts) < cooldown_seconds
