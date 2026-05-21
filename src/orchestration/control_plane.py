"""ControlPlane wires StateStore + EventBus + LocalDockerAgent for a Scheduler."""
from orchestration.agent import LocalDockerAgent
from orchestration.bus import EventBus
from orchestration.scheduler import Scheduler
from orchestration.state import FlowRun, InMemoryStateStore
from orchestration.types import Flow


class ControlPlane:
    """Top-level facade: owns the store/bus/agent and runs flows end-to-end."""

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

    async def run_flow(self, flow: Flow) -> FlowRun:
        """Create a FlowRun for `flow` and drive it to completion via a fresh Scheduler."""
        run = self.store.create_flow_run(
            flow.name, [s.name for s in flow.steps()],
        )
        scheduler = Scheduler(flow, run, self.store, self.agent, self.bus)
        await scheduler.run_until_done()
        return run
