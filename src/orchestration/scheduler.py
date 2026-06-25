import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from orchestration.agent import LaunchSpec, LocalDockerAgent
from orchestration.bus import EventBus
from orchestration.events import (
    ContainerExited,
    ContainerStarted,
    Event,
    FlowStateChanged,
    StepStateChanged,
)
from orchestration.state import (
    FlowRun,
    FlowState,
    InMemoryStateStore,
    StepState,
)
from orchestration.types import Flow, Step, SubFlow

if TYPE_CHECKING:
    from orchestration.control_plane import ControlPlane


@dataclass(frozen=True)
class _SubFlowDone:
    """Internal inbox signal: a SubFlow node's child run reached a terminal state."""

    step_name: str
    succeeded: bool


class Scheduler:
    """Drives a single FlowRun to completion."""

    def __init__(
        self,
        flow: Flow,
        run: FlowRun,
        store: InMemoryStateStore,
        agent: LocalDockerAgent,
        bus: EventBus,
        control_plane: "ControlPlane",
    ) -> None:
        """Bind the flow definition, run, store, agent, bus, and owning control plane."""
        self.flow = flow
        self.run = run
        self.store = store
        self.agent = agent
        self.bus = bus
        self.cp = control_plane
        self._container_to_step: dict[str, str] = {}
        self._inbox: asyncio.Queue[object] = asyncio.Queue()
        self._subflow_tasks: dict[str, asyncio.Task[FlowState]] = {}

    def feed(self, event: Event) -> None:
        """Deliver an agent event belonging to this run (called by the router)."""
        self._inbox.put_nowait(event)

    async def run_until_done(self) -> FlowState:
        """Main loop: launch ready work, drain the inbox, return final flow state."""
        await self._transition_flow(FlowState.pending, FlowState.running)
        await self._advance()

        while self.run.state == FlowState.running:
            item = await self._inbox.get()
            if isinstance(item, _SubFlowDone):
                await self._on_subflow_done(item)
                continue
            await self._publish(item)
            if isinstance(item, ContainerStarted):
                await self._on_container_started(item)
            elif isinstance(item, ContainerExited):
                await self._on_container_exited(item)
        self._cancel_inflight_subflows()
        return self.run.state

    def _cancel_inflight_subflows(self) -> None:
        """On terminal flow state, cancel any subflow whose child run is still going."""
        for task in self._subflow_tasks.values():
            task.cancel()
        self._subflow_tasks.clear()

    _TERMINAL = frozenset(
        {StepState.succeeded, StepState.failed, StepState.skipped}
    )

    async def _advance(self) -> None:
        """Launch or skip every newly-resolved pending node, cascading skips."""
        changed = True
        while changed:
            changed = False
            for name in self.flow.topological_order():
                if self.run.steps[name].state != StepState.pending:
                    continue
                deps = self.flow.deps(name)
                if any(self.run.steps[d].state not in self._TERMINAL for d in deps):
                    continue
                if any(self.run.steps[d].state == StepState.failed for d in deps):
                    continue
                if self._should_skip(name, deps):
                    await self._transition_step(
                        name, StepState.pending, StepState.skipped
                    )
                else:
                    await self._launch_step(name)
                changed = True

    def _should_skip(self, step_name: str, deps: tuple[str, ...]) -> bool:
        """Skip if any dep was skipped, or the node's `when` condition is False."""
        if any(self.run.steps[d].state == StepState.skipped for d in deps):
            return True
        condition = self.flow.condition(step_name)
        return condition is not None and not condition(self.run.steps)

    async def _launch_step(self, step_name: str) -> None:
        """Mark a node scheduled and launch it as a container or a nested subflow."""
        work = self.flow.work(step_name)
        await self._transition_step(step_name, StepState.pending, StepState.scheduled)
        if isinstance(work, SubFlow):
            await self._launch_subflow(step_name, work)
        else:
            await self._launch_container(step_name, work)

    async def _launch_container(self, step_name: str, step: Step) -> None:
        """Ask the agent to launch a single-container step."""
        spec = LaunchSpec(
            flow_run_id=self.run.id,
            step_name=step_name,
            image=step.image,
            command=step.command,
            env=dict(step.env),
            resources=step.resources,
            volumes=dict(step.volumes),
            workdir=step.workdir,
        )
        cid = await self.agent.launch(spec)
        self._container_to_step[cid] = step_name

    async def _launch_subflow(self, step_name: str, sub: SubFlow) -> None:
        """Spawn a child run for a SubFlow node; signal the inbox when it finishes."""
        await self._transition_step(step_name, StepState.scheduled, StepState.running)
        task: asyncio.Task[FlowState] = asyncio.create_task(
            self.cp.run_subflow(sub.build(), self.run.id, step_name)
        )

        def _on_done(t: "asyncio.Task[FlowState]", name: str = step_name) -> None:
            """Translate child-run completion into an inbox signal for this loop."""
            succeeded = (
                not t.cancelled()
                and t.exception() is None
                and t.result() == FlowState.succeeded
            )
            self._inbox.put_nowait(_SubFlowDone(step_name=name, succeeded=succeeded))

        task.add_done_callback(_on_done)
        self._subflow_tasks[step_name] = task

    async def _on_subflow_done(self, signal: _SubFlowDone) -> None:
        """Advance a SubFlow node from running on child success; fail the flow otherwise."""
        self._subflow_tasks.pop(signal.step_name, None)
        if signal.succeeded:
            await self._transition_step(
                signal.step_name, StepState.running, StepState.succeeded
            )
            await self._advance()
            await self._maybe_finish_flow()
        else:
            await self._transition_step(
                signal.step_name, StepState.running, StepState.failed
            )
            await self._transition_flow(FlowState.running, FlowState.failed)

    async def _on_container_started(self, event: ContainerStarted) -> None:
        """Persist the event and move the step from scheduled to running."""
        self.store.append(event)
        await self._transition_step(
            event.step_name, StepState.scheduled, StepState.running
        )

    async def _on_container_exited(self, event: ContainerExited) -> None:
        """Treat an expected exit code as success (advancing); else fail the flow."""
        self.store.append(event)
        step = self.flow.work(event.step_name)
        expected = getattr(step, "expected_exit_codes", frozenset({0}))
        if event.exit_code in expected:
            await self._transition_step(
                event.step_name, StepState.running, StepState.succeeded
            )
            await self._advance()
            await self._maybe_finish_flow()
        else:
            await self._transition_step(
                event.step_name, StepState.running, StepState.failed
            )
            await self._transition_flow(FlowState.running, FlowState.failed)

    async def _maybe_finish_flow(self) -> None:
        """Succeed the flow once every node is terminal and none failed (skips are OK)."""
        states = {sr.state for sr in self.run.steps.values()}
        if states <= {StepState.succeeded, StepState.skipped}:
            await self._transition_flow(FlowState.running, FlowState.succeeded)

    async def _transition_step(
        self, step_name: str, from_state: StepState, to_state: StepState
    ) -> None:
        """Idempotently record a step state transition; no-op on mismatched from_state."""
        if self.run.steps[step_name].state != from_state:
            return
        evt = StepStateChanged(
            flow_run_id=self.run.id,
            step_name=step_name,
            from_state=from_state.value,
            to_state=to_state.value,
        )
        self.store.append(evt)
        await self._publish(evt)

    async def _transition_flow(
        self, from_state: FlowState, to_state: FlowState
    ) -> None:
        """Idempotently record a flow state transition; no-op on mismatched from_state."""
        if self.run.state != from_state:
            return
        evt = FlowStateChanged(
            flow_run_id=self.run.id,
            from_state=from_state.value,
            to_state=to_state.value,
        )
        self.store.append(evt)
        await self._publish(evt)

    async def _publish(self, event: Event) -> None:
        """Forward an event to the EventBus."""
        await self.bus.publish(event)
