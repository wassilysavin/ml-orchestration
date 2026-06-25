import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Protocol

from orchestration.events import (
    ContainerExited,
    ContainerStarted,
    Event,
    FlowRunCreated,
    FlowStateChanged,
    LogChunk,
    StepStateChanged,
)

_PERSISTED: dict[str, type] = {
    "FlowRunCreated": FlowRunCreated,
    "FlowStateChanged": FlowStateChanged,
    "StepStateChanged": StepStateChanged,
    "ContainerStarted": ContainerStarted,
    "ContainerExited": ContainerExited,
}


@dataclass
class StepView:
    """Materialized state of one node (container step or SubFlow node)."""

    name: str
    state: str = "pending"
    container_id: str | None = None
    host_id: str | None = None
    exit_code: int | None = None
    history: list[tuple[float, str, str]] = field(default_factory=list)


@dataclass
class RunView:
    """Materialized state of one flow run, including its parent linkage."""

    id: str
    flow_name: str
    state: str = "pending"
    parent_flow_run_id: str | None = None
    parent_step_name: str | None = None
    steps: dict[str, StepView] = field(default_factory=dict)
    created_ts: float | None = None
    updated_ts: float | None = None


class Sink(Protocol):
    """Durable append target for persisted monitoring events."""

    def append(self, record: dict) -> None:
        """Persist one serialized event record."""
        ...


class NullSink:
    """Drop-everything sink for in-memory-only monitoring."""

    def append(self, record: dict) -> None:
        """Discard the record."""


class JsonlSink:
    """Append-only JSONL file sink (one event record per line)."""

    def __init__(self, path: str) -> None:
        """Open `path` for appending; each append writes and flushes one line."""
        self._fh = open(path, "a", encoding="utf-8")

    def append(self, record: dict) -> None:
        """Write `record` as a JSON line and flush so a crash loses nothing."""
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self._fh.close()


class MonitoringService:
    """Consumes control-plane events into a queryable, persistable run tree."""

    def __init__(self, sink: Sink | None = None) -> None:
        """Start with an empty view and an optional durable sink (default: none)."""
        self._runs: dict[str, RunView] = {}
        self._sink = sink or NullSink()

    def handle(self, event: Event) -> None:
        """EventBus handler: persist then apply each state-bearing event."""
        type_name = type(event).__name__
        if type_name not in _PERSISTED:
            return
        self._sink.append({"type": type_name, "data": asdict(event)})
        self._apply(event)

    def _apply(self, event: Event) -> None:
        """Fold one event into the materialized view."""
        if isinstance(event, FlowRunCreated):
            self._runs[event.flow_run_id] = RunView(
                id=event.flow_run_id,
                flow_name=event.flow_name,
                parent_flow_run_id=event.parent_flow_run_id,
                parent_step_name=event.parent_step_name,
                steps={n: StepView(name=n) for n in event.step_names},
                created_ts=event.ts,
                updated_ts=event.ts,
            )
        elif isinstance(event, FlowStateChanged):
            run = self._runs[event.flow_run_id]
            run.state = event.to_state
            run.updated_ts = event.ts
        elif isinstance(event, StepStateChanged):
            step = self._runs[event.flow_run_id].steps[event.step_name]
            step.state = event.to_state
            step.history.append((event.ts, event.from_state, event.to_state))
            self._runs[event.flow_run_id].updated_ts = event.ts
        elif isinstance(event, ContainerStarted):
            step = self._runs[event.flow_run_id].steps[event.step_name]
            step.container_id = event.container_id
            step.host_id = event.host_id
        elif isinstance(event, ContainerExited):
            self._runs[event.flow_run_id].steps[event.step_name].exit_code = (
                event.exit_code
            )

    def run(self, run_id: str) -> RunView:
        """Return the materialized view of a single run."""
        return self._runs[run_id]

    def runs(self) -> list[RunView]:
        """Return every known run view."""
        return list(self._runs.values())

    def roots(self) -> list[RunView]:
        """Return top-level runs (those with no parent)."""
        return [r for r in self._runs.values() if r.parent_flow_run_id is None]

    def _children_by_step(self, run_id: str) -> dict[str, list[str]]:
        """Map each step name of `run_id` to the child run ids it spawned."""
        out: dict[str, list[str]] = defaultdict(list)
        for r in self._runs.values():
            if r.parent_flow_run_id == run_id and r.parent_step_name is not None:
                out[r.parent_step_name].append(r.id)
        return out

    def tree(self, run_id: str) -> dict:
        """Return a nested dict of the run, its steps, and their subflow runs."""
        run = self._runs[run_id]
        children = self._children_by_step(run_id)
        steps = []
        for name, sv in run.steps.items():
            node: dict = {"step": name, "state": sv.state}
            if sv.exit_code is not None:
                node["exit_code"] = sv.exit_code
            subs = [self.tree(cid) for cid in children.get(name, [])]
            if subs:
                node["subflows"] = subs
            steps.append(node)
        return {
            "run_id": run.id,
            "flow": run.flow_name,
            "state": run.state,
            "steps": steps,
        }

    def summary(self) -> dict[str, int]:
        """Return counts of steps across all runs grouped by state."""
        counts: dict[str, int] = defaultdict(int)
        for r in self._runs.values():
            for s in r.steps.values():
                counts[s.state] += 1
        return dict(counts)

    def render_tree(self, run_id: str) -> str:
        """Render the run tree as an indented, human-readable string."""
        lines: list[str] = []

        def walk(node: dict, depth: int) -> None:
            """Append one run and recurse into its steps/subflows."""
            pad = "  " * depth
            lines.append(f"{pad}{node['flow']} [{node['state']}] ({node['run_id']})")
            for step in node["steps"]:
                code = (
                    f" exit={step['exit_code']}" if "exit_code" in step else ""
                )
                lines.append(f"{pad}  - {step['step']}: {step['state']}{code}")
                for sub in step.get("subflows", []):
                    walk(sub, depth + 2)

        walk(self.tree(run_id), 0)
        return "\n".join(lines)

    @classmethod
    def replay(cls, records: Iterable[dict]) -> "MonitoringService":
        """Rebuild a view (no sink) from previously persisted event records."""
        service = cls()
        for record in records:
            event_cls = _PERSISTED.get(record["type"])
            if event_cls is None:
                continue
            data = dict(record["data"])
            if "step_names" in data:
                data["step_names"] = tuple(data["step_names"])
            service._apply(event_cls(**data))
        return service

    @classmethod
    def replay_file(cls, path: str) -> "MonitoringService":
        """Rebuild a view from a JSONL log written by JsonlSink."""
        with open(path, encoding="utf-8") as fh:
            return cls.replay(json.loads(line) for line in fh if line.strip())
