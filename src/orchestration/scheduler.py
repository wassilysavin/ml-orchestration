"""Per-FlowRun scheduler loop driving steps from pending to terminal state."""
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
from orchestration.types import Flow


class Scheduler:
    """Drives a single FlowRun to completion."""

    def __init__(
        self,
        flow: Flow,
        run: FlowRun,
        store: InMemoryStateStore,
        agent: LocalDockerAgent,
        bus: EventBus,
    ) -> None:
        """Bind the flow definition, materialized run, store, agent, and event bus."""
        self.flow = flow
        self.run = run
        self.store = store
        self.agent = agent
        self.bus = bus
        self._container_to_step: dict[str, str] = {}

    async def run_until_done(self) -> FlowState:
        """Main loop: launch ready steps, drain agent events, return final flow state."""
        await self._transition_flow(FlowState.pending, FlowState.running)
        await self._launch_ready()

        agent_q = self.agent.events()
        while self.run.state == FlowState.running:
            event = await agent_q.get()
            await self._publish(event)
            if isinstance(event, ContainerStarted):
                await self._on_container_started(event)
            elif isinstance(event, ContainerExited):
                await self._on_container_exited(event)
        return self.run.state

    async def _launch_ready(self) -> None:
        """Launch every pending step whose dependencies are all succeeded."""
        for name in self.flow.topological_order():
            if self.run.steps[name].state != StepState.pending:
                continue
            if not self._deps_satisfied(name):
                continue
            await self._launch_step(name)

    def _deps_satisfied(self, step_name: str) -> bool:
        """Return True iff every dependency of `step_name` is in StepState.succeeded."""
        for dep in self.flow.deps(step_name):
            if self.run.steps[dep].state != StepState.succeeded:
                return False
        return True

    async def _launch_step(self, step_name: str) -> None:
        """Mark a step scheduled and ask the agent to launch its container."""
        step = next(s for s in self.flow.steps() if s.name == step_name)
        await self._transition_step(step_name, StepState.pending, StepState.scheduled)
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

    async def _on_container_started(self, event: ContainerStarted) -> None:
        """Persist the event and move the step from scheduled to running."""
        self.store.append(event)
        await self._transition_step(
            event.step_name, StepState.scheduled, StepState.running
        )

    async def _on_container_exited(self, event: ContainerExited) -> None:
        """On clean exit advance to succeeded + relaunch; on non-zero fail the flow."""
        self.store.append(event)
        if event.exit_code == 0:
            await self._transition_step(
                event.step_name, StepState.running, StepState.succeeded
            )
            await self._launch_ready()
            await self._maybe_finish_flow()
        else:
            await self._transition_step(
                event.step_name, StepState.running, StepState.failed
            )
            await self._transition_flow(FlowState.running, FlowState.failed)

    async def _maybe_finish_flow(self) -> None:
        """Transition the flow to succeeded once every step has succeeded."""
        states = {sr.state for sr in self.run.steps.values()}
        if states == {StepState.succeeded}:
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
