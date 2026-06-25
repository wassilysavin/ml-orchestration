import enum
import uuid
from dataclasses import dataclass, field
from typing import Iterable

from orchestration.events import (
    ContainerExited,
    ContainerStarted,
    Event,
    FlowStateChanged,
    StepStateChanged,
)


class FlowState(str, enum.Enum):
    """Lifecycle states a FlowRun can occupy."""

    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


class StepState(str, enum.Enum):
    """Lifecycle states an individual StepRun can occupy."""

    pending = "pending"
    scheduled = "scheduled"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    skipped = "skipped"


@dataclass
class StepRun:
    """Materialized per-step state: container id, host, current state, exit code."""

    name: str
    state: StepState = StepState.pending
    container_id: str | None = None
    host_id: str | None = None
    exit_code: int | None = None


@dataclass
class FlowRun:
    """Materialized per-flow state: id, name, current state, and its StepRuns."""

    id: str
    flow_name: str
    state: FlowState = FlowState.pending
    steps: dict[str, StepRun] = field(default_factory=dict)
    parent_flow_run_id: str | None = None
    parent_step_name: str | None = None


def new_flow_run_id() -> str:
    """Return a fresh 12-hex-char flow-run identifier."""
    return uuid.uuid4().hex[:12]


class InMemoryStateStore:
    """Append-only event log + materialized FlowRun view."""

    def __init__(self) -> None:
        """Create empty run table and event log."""
        self._runs: dict[str, FlowRun] = {}
        self._events: list[Event] = []

    def create_flow_run(
        self,
        flow_name: str,
        step_names: Iterable[str],
        parent_flow_run_id: str | None = None,
        parent_step_name: str | None = None,
    ) -> FlowRun:
        """Allocate a new FlowRun with `pending` step entries for each step name."""
        run = FlowRun(
            id=new_flow_run_id(),
            flow_name=flow_name,
            steps={n: StepRun(name=n) for n in step_names},
            parent_flow_run_id=parent_flow_run_id,
            parent_step_name=parent_step_name,
        )
        self._runs[run.id] = run
        return run

    def children_of(self, flow_run_id: str) -> list[FlowRun]:
        """Return the child runs spawned by SubFlow nodes of `flow_run_id`."""
        return [
            r for r in self._runs.values() if r.parent_flow_run_id == flow_run_id
        ]

    def append(self, event: Event) -> None:
        """Record `event` to the log and apply it to the materialized view."""
        self._events.append(event)
        self._apply(event)

    def _apply(self, event: Event) -> None:
        """Mutate the relevant FlowRun/StepRun fields based on `event`'s type."""
        if isinstance(event, FlowStateChanged):
            self._runs[event.flow_run_id].state = FlowState(event.to_state)
        elif isinstance(event, StepStateChanged):
            self._runs[event.flow_run_id].steps[event.step_name].state = StepState(
                event.to_state
            )
        elif isinstance(event, ContainerStarted):
            step = self._runs[event.flow_run_id].steps[event.step_name]
            step.container_id = event.container_id
            step.host_id = event.host_id
        elif isinstance(event, ContainerExited):
            step = self._runs[event.flow_run_id].steps[event.step_name]
            step.exit_code = event.exit_code

    def get(self, flow_run_id: str) -> FlowRun:
        """Look up a FlowRun by id."""
        return self._runs[flow_run_id]
