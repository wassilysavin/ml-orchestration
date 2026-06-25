import time
from dataclasses import dataclass, field
from typing import Literal


def _now() -> float:
    """Return the current wall-clock time as a Unix timestamp."""
    return time.time()


@dataclass(frozen=True)
class FlowRunCreated:
    """Emitted when a FlowRun is allocated, before it starts running."""

    flow_run_id: str
    flow_name: str
    step_names: tuple[str, ...]
    parent_flow_run_id: str | None = None
    parent_step_name: str | None = None
    ts: float = field(default_factory=_now)


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
    FlowRunCreated
    | FlowStateChanged
    | StepStateChanged
    | ContainerStarted
    | ContainerExited
    | LogChunk
)
