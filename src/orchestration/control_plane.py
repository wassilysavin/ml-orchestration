import asyncio

from orchestration.agent import LocalDockerAgent
from orchestration.bus import EventBus
from orchestration.events import FlowRunCreated
from orchestration.scheduler import Scheduler
from orchestration.state import FlowRun, FlowState, InMemoryStateStore
from orchestration.types import Flow


class ControlPlane:
    """Top-level facade: owns the store/bus/agent and runs flows (and subflows)."""

    def __init__(
        self,
        agent: LocalDockerAgent,
        store: InMemoryStateStore | None = None,
        bus: EventBus | None = None,
    ) -> None:
        """Wire in an agent and optional pre-built store/bus (else defaults)."""
        self.store = store or InMemoryStateStore()
        self.bus = bus or EventBus()
        self.agent = agent
        self._schedulers: dict[str, Scheduler] = {}
        self._pump_task: asyncio.Task[None] | None = None

    async def run_flow(
        self,
        flow: Flow,
        parent_flow_run_id: str | None = None,
        parent_step_name: str | None = None,
    ) -> FlowRun:
        """Create a FlowRun for `flow` and drive it to completion."""
        run = self.store.create_flow_run(
            flow.name,
            [s.name for s in flow.steps()],
            parent_flow_run_id=parent_flow_run_id,
            parent_step_name=parent_step_name,
        )
        await self.bus.publish(
            FlowRunCreated(
                flow_run_id=run.id,
                flow_name=run.flow_name,
                step_names=tuple(run.steps),
                parent_flow_run_id=parent_flow_run_id,
                parent_step_name=parent_step_name,
            )
        )
        scheduler = Scheduler(flow, run, self.store, self.agent, self.bus, self)
        self._schedulers[run.id] = scheduler

        owns_pump = self._pump_task is None
        if owns_pump:
            self._pump_task = asyncio.create_task(self._pump())
        try:
            await scheduler.run_until_done()
        finally:
            self._schedulers.pop(run.id, None)
            if owns_pump:
                await self._stop_pump()
        return run

    async def run_subflow(
        self, child_flow: Flow, parent_flow_run_id: str, parent_step_name: str
    ) -> FlowState:
        """Run a SubFlow node's child flow under the running router; return its state."""
        run = await self.run_flow(child_flow, parent_flow_run_id, parent_step_name)
        return run.state

    async def _pump(self) -> None:
        """Drain the shared agent queue, routing each event to its owning scheduler."""
        queue = self.agent.events()
        while True:
            event = await queue.get()
            scheduler = self._schedulers.get(event.flow_run_id)
            if scheduler is not None:
                scheduler.feed(event)

    async def _stop_pump(self) -> None:
        """Cancel and await the router task once the top-level flow is done."""
        assert self._pump_task is not None
        self._pump_task.cancel()
        try:
            await self._pump_task
        except asyncio.CancelledError:
            pass
        self._pump_task = None
