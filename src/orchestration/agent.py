import asyncio
from dataclasses import dataclass, field
from typing import Any

from orchestration.events import (
    ContainerExited,
    ContainerStarted,
    Event,
    LogChunk,
)
from orchestration.types import Resources


@dataclass
class LaunchSpec:
    """Everything the agent needs to start one step's container."""

    flow_run_id: str
    step_name: str
    image: str
    command: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    resources: Resources = Resources()
    volumes: dict[str, str] = field(default_factory=dict)
    workdir: str | None = None


class LocalDockerAgent:
    """Drives the local docker daemon and surfaces container lifecycle events."""

    def __init__(self, host_id: str = "host-local") -> None:
        """Open a docker client and prepare the event queue + monitor registry."""
        import docker
        self.host_id = host_id
        self._client = docker.from_env()
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._monitors: dict[str, asyncio.Task[None]] = {}

    def events(self) -> asyncio.Queue[Event]:
        """Return the queue the scheduler drains for container lifecycle events."""
        return self._queue

    async def launch(self, spec: LaunchSpec) -> str:
        """Start a detached container per `spec`, emit ContainerStarted, spawn monitor."""
        volumes = {
            host: {"bind": container, "mode": "rw"}
            for host, container in spec.volumes.items()
        } or None
        environment = {**dict(spec.env), "FLOW_RUN_ID": spec.flow_run_id}
        container = await asyncio.to_thread(
            self._client.containers.run,
            image=spec.image,
            command=spec.command,
            environment=environment,
            volumes=volumes,
            working_dir=spec.workdir,
            detach=True,
            labels={
                "orchestration.flow_run_id": spec.flow_run_id,
                "orchestration.step_name": spec.step_name,
                "orchestration.host_id": self.host_id,
            },
            nano_cpus=int(spec.resources.cpu * 1e9),
            mem_limit=f"{int(spec.resources.memory_gb * 1024)}m",
        )
        cid = container.id
        await self._queue.put(ContainerStarted(
            flow_run_id=spec.flow_run_id,
            step_name=spec.step_name,
            container_id=cid,
            host_id=self.host_id,
        ))
        task = asyncio.create_task(
            self._monitor(spec.flow_run_id, spec.step_name, cid)
        )
        self._monitors[cid] = task
        return cid

    async def kill(self, container_id: str) -> None:
        """Best-effort kill; the monitor will still surface the exit event."""
        try:
            container = await asyncio.to_thread(
                self._client.containers.get, container_id
            )
            await asyncio.to_thread(container.kill)
        except Exception:
            pass

    async def _monitor(
        self, flow_run_id: str, step_name: str, container_id: str
    ) -> None:
        """Stream logs then wait for exit; always emits a ContainerExited."""
        try:
            await self._stream_logs(flow_run_id, step_name, container_id)
            exit_code = await asyncio.to_thread(self._wait_for_exit, container_id)
            await self._queue.put(ContainerExited(
                flow_run_id=flow_run_id,
                step_name=step_name,
                container_id=container_id,
                exit_code=exit_code,
            ))
        except asyncio.CancelledError:
            raise
        except Exception:
            await self._queue.put(ContainerExited(
                flow_run_id=flow_run_id,
                step_name=step_name,
                container_id=container_id,
                exit_code=-1,
            ))
            raise
        finally:
            self._monitors.pop(container_id, None)

    async def _stream_logs(
        self, flow_run_id: str, step_name: str, container_id: str
    ) -> None:
        """Pump container stdout/stderr chunks onto the agent queue as LogChunks."""
        container = await asyncio.to_thread(
            self._client.containers.get, container_id
        )
        loop = asyncio.get_running_loop()
        queue = self._queue

        def pump() -> None:
            """Blocking inner loop that runs in a worker thread and feeds the asyncio queue."""
            try:
                for chunk in container.logs(stream=True, follow=True, stdout=True, stderr=True):
                    if not chunk:
                        continue
                    evt = LogChunk(
                        flow_run_id=flow_run_id,
                        step_name=step_name,
                        container_id=container_id,
                        stream="stdout",
                        data=chunk if isinstance(chunk, bytes) else bytes(chunk),
                    )
                    asyncio.run_coroutine_threadsafe(queue.put(evt), loop)
            except Exception:
                pass

        await asyncio.to_thread(pump)

    def _wait_for_exit(self, container_id: str) -> int:
        """Block until the container exits; return its integer status code."""
        container = self._client.containers.get(container_id)
        result: dict[str, Any] = container.wait()
        return int(result.get("StatusCode", -1))

    async def shutdown(self) -> None:
        """Cancel every monitor task and close the docker client."""
        for task in list(self._monitors.values()):
            task.cancel()
        for task in list(self._monitors.values()):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._monitors.clear()
        try:
            self._client.close()
        except Exception:
            pass
