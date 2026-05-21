"""Frozen event dataclasses flowing through the EventBus and agent queue."""
import time
from dataclasses import dataclass, field
from typing import Literal


def _now() -> float:
    """Return the current wall-clock time as a Unix timestamp."""
    return time.time()


@dataclass(frozen=True)
class FlowStateChanged:
    """Emitted when a FlowRun transitions between FlowState values."""

    flow_run_id: str
    from_state: str
    to_state: str
    ts: float = field(default_factory=_now)


@dataclass(frozen=True)
class StepStateChanged:
    """Emitted when a StepRun transitions between StepState values."""

    flow_run_id: str
    step_name: str
    from_state: str
    to_state: str
    ts: float = field(default_factory=_now)


@dataclass(frozen=True)
class ContainerStarted:
    """Emitted by the agent once a Docker container exists."""

    flow_run_id: str
    step_name: str
    container_id: str
    host_id: str
    ts: float = field(default_factory=_now)


@dataclass(frozen=True)
class ContainerExited:
    """Emitted by the agent once a Docker container has exited, with its status code."""

    flow_run_id: str
    step_name: str
    container_id: str
    exit_code: int
    ts: float = field(default_factory=_now)


@dataclass(frozen=True)
class LogChunk:
    """A raw chunk of container stdout/stderr from the agent's log pump."""

    flow_run_id: str
    step_name: str
    container_id: str
    stream: Literal["stdout", "stderr"]
    data: bytes
    ts: float = field(default_factory=_now)


Event = (
    FlowStateChanged
    | StepStateChanged
    | ContainerStarted
    | ContainerExited
    | LogChunk
)
