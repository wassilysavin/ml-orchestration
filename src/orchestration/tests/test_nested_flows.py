import asyncio

import pytest

from orchestration.control_plane import ControlPlane
from orchestration.events import ContainerExited, ContainerStarted, Event
from orchestration.state import FlowState, StepState
from orchestration.types import Flow, Resources, Step, SubFlow, exited_with

NO_DRIFT = 64


class FakeAgent:
    """In-memory stand-in for LocalDockerAgent: no Docker, deterministic exits."""

    def __init__(
        self,
        failing_images: set[str] | None = None,
        exit_codes: dict[str, int] | None = None,
    ) -> None:
        """Record per-image exit behaviour; prepare queue + counters."""
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._failing = failing_images or set()
        self._exit_codes = exit_codes or {}
        self._n = 0
        self.launched: list[tuple[str, str]] = []
        self.max_concurrent = 0
        self._running = 0

    def _code_for(self, image: str) -> int:
        """Resolve the exit code an image should produce."""
        if image in self._exit_codes:
            return self._exit_codes[image]
        return 1 if image in self._failing else 0

    def events(self) -> asyncio.Queue[Event]:
        """Return the lifecycle event queue the control-plane router drains."""
        return self._queue

    async def launch(self, spec) -> str:
        """Emit ContainerStarted now and schedule ContainerExited asynchronously."""
        self._n += 1
        cid = f"c{self._n}"
        self.launched.append((spec.image, spec.step_name))
        self._running += 1
        self.max_concurrent = max(self.max_concurrent, self._running)
        await self._queue.put(
            ContainerStarted(
                flow_run_id=spec.flow_run_id,
                step_name=spec.step_name,
                container_id=cid,
                host_id="fake",
            )
        )
        asyncio.create_task(self._finish(spec, cid, self._code_for(spec.image)))
        return cid

    async def _finish(self, spec, cid: str, exit_code: int) -> None:
        """After yielding control, surface the container's exit event."""
        await asyncio.sleep(0)
        self._running -= 1
        await self._queue.put(
            ContainerExited(
                flow_run_id=spec.flow_run_id,
                step_name=spec.step_name,
                container_id=cid,
                exit_code=exit_code,
            )
        )


class _Img(Step):
    """Minimal container step; image set per-instance to vary success/failure."""

    image = "noop:latest"

    def __init__(self, name: str, image: str = "noop:latest") -> None:
        """Allow per-instance image + name (Step normally takes name only)."""
        self.image = image
        super().__init__(name=name)


def _training_subflow(tag: str, image: str = "noop:latest") -> SubFlow:
    """Build a SubFlow node whose child flow is a train -> robustness chain."""

    def build() -> Flow:
        f = Flow(f"train-{tag}")
        train = f.add(_Img(f"train-{tag}", image))
        f.add(_Img(f"robustness-{tag}"), after=train)
        return f

    return SubFlow(build, name=f"hp-{tag}")


def test_parallel_subflows_per_hyperparameter() -> None:
    """A parent flow fans out N identical training subflows in parallel."""
    agent = FakeAgent()
    cp = ControlPlane(agent)

    flow = Flow("sweep")
    prep = flow.add(_Img("prep"))
    tags = ["a", "b", "c"]
    subs = [flow.add(_training_subflow(t), after=prep) for t in tags]
    flow.add(_Img("gather"), after=subs)

    run = asyncio.run(cp.run_flow(flow))

    assert run.state == FlowState.succeeded
    for t in tags:
        assert run.steps[f"hp-{t}"].state == StepState.succeeded

    children = cp.store.children_of(run.id)
    assert len(children) == 3
    assert all(c.parent_flow_run_id == run.id for c in children)
    assert all(c.state == FlowState.succeeded for c in children)

    assert agent.max_concurrent >= 3
    assert len(agent.launched) == 8


def test_subflow_failure_fails_parent() -> None:
    """A non-zero exit inside one subflow fails that node and the parent flow."""
    agent = FakeAgent(failing_images={"boom:latest"})
    cp = ControlPlane(agent)

    flow = Flow("sweep")
    flow.add(_training_subflow("ok"))
    flow.add(_training_subflow("bad", image="boom:latest"))

    run = asyncio.run(cp.run_flow(flow))

    assert run.state == FlowState.failed
    assert run.steps["hp-bad"].state == StepState.failed


class _Drift(Step):
    """A decision step: exit 0 means 'drift detected', NO_DRIFT means 'skip retrain'."""

    image = "drift:latest"
    expected_exit_codes = frozenset({0, NO_DRIFT})


def _conditional_flow() -> Flow:
    """drift -> (retrain -> promote) only when drift exited 0 (drift detected)."""
    flow = Flow("monitor")
    drift = flow.add(_Drift("drift"))
    retrain = flow.add(_Img("retrain"), after=drift, when=exited_with(drift, 0))
    flow.add(_Img("promote"), after=retrain)
    return flow


def test_no_drift_skips_retrain_subtree() -> None:
    """When the drift step signals NO_DRIFT, retrain and promote are skipped, flow OK."""
    agent = FakeAgent(exit_codes={"drift:latest": NO_DRIFT})
    cp = ControlPlane(agent)

    run = asyncio.run(cp.run_flow(_conditional_flow()))

    assert run.state == FlowState.succeeded
    assert run.steps["drift"].state == StepState.succeeded
    assert run.steps["retrain"].state == StepState.skipped
    assert run.steps["promote"].state == StepState.skipped
    assert [img for img, _ in agent.launched] == ["drift:latest"]


def test_drift_detected_runs_retrain_subtree() -> None:
    """When the drift step exits 0, the retrain subtree runs normally."""
    agent = FakeAgent(exit_codes={"drift:latest": 0})
    cp = ControlPlane(agent)

    run = asyncio.run(cp.run_flow(_conditional_flow()))

    assert run.state == FlowState.succeeded
    assert run.steps["retrain"].state == StepState.succeeded
    assert run.steps["promote"].state == StepState.succeeded


def test_skipped_subflow_node() -> None:
    """A SubFlow node gated by a false condition is skipped without spawning a child."""
    agent = FakeAgent(exit_codes={"drift:latest": NO_DRIFT})
    cp = ControlPlane(agent)

    flow = Flow("monitor-retrain")
    drift = flow.add(_Drift("drift"))
    flow.add(_training_subflow("x"), after=drift, when=exited_with(drift, 0))

    run = asyncio.run(cp.run_flow(flow))

    assert run.state == FlowState.succeeded
    assert run.steps["hp-x"].state == StepState.skipped
    assert cp.store.children_of(run.id) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
