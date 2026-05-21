"""Typer CLI: `orchestration run path/to/file.py:flow`."""
import asyncio
import importlib
import importlib.util
import sys
from pathlib import Path

import typer
from rich.console import Console

from orchestration.agent import LocalDockerAgent
from orchestration.control_plane import ControlPlane
from orchestration.events import (
    ContainerExited,
    ContainerStarted,
    Event,
    FlowStateChanged,
    LogChunk,
    StepStateChanged,
)
from orchestration.state import FlowState
from orchestration.types import Flow


app = typer.Typer(add_completion=False, help="ml-orchestration CLI")
console = Console()


def _load_flow(target: str) -> Flow:
    """Load `path/to/file.py:var` or `pkg.module:var`."""
    if ":" not in target:
        raise typer.BadParameter("expected 'path_or_module:var'")
    mod_part, var = target.split(":", 1)

    if mod_part.endswith(".py") or "/" in mod_part:
        path = Path(mod_part).resolve()
        if not path.exists():
            raise typer.BadParameter(f"file not found: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise typer.BadParameter(f"could not load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(path.parent))
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(mod_part)

    if not hasattr(module, var):
        raise typer.BadParameter(f"{mod_part} has no attribute '{var}'")
    flow = getattr(module, var)
    if not isinstance(flow, Flow):
        raise typer.BadParameter(f"{target} is not a Flow")
    return flow


def _render_event(event: Event) -> str | None:
    """Format an Event as a Rich-markup line, or None to skip rendering."""
    if isinstance(event, FlowStateChanged):
        return f"[bold]flow[/] {event.from_state} → [cyan]{event.to_state}[/]"
    if isinstance(event, StepStateChanged):
        color = {
            "scheduled": "yellow",
            "running": "blue",
            "succeeded": "green",
            "failed": "red",
        }.get(event.to_state, "white")
        return (
            f"  step [bold]{event.step_name}[/] "
            f"{event.from_state} → [{color}]{event.to_state}[/]"
        )
    if isinstance(event, ContainerStarted):
        return (
            f"    [dim]container {event.container_id[:12]} started on "
            f"{event.host_id} ({event.step_name})[/]"
        )
    if isinstance(event, ContainerExited):
        return (
            f"    [dim]container {event.container_id[:12]} exited "
            f"code={event.exit_code} ({event.step_name})[/]"
        )
    if isinstance(event, LogChunk):
        try:
            text = event.data.decode("utf-8", errors="replace").rstrip()
        except Exception:
            return None
        if not text:
            return None
        return f"      [grey50]{event.step_name}|[/] {text}"
    return None


@app.command()
def run(
    target: str = typer.Argument(..., help="flow target as path.py:var or module:var"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="suppress event stream"),
) -> None:
    """Load a flow target, run it under a fresh ControlPlane, exit with its status."""
    flow = _load_flow(target)

    async def _go() -> int:
        """Async body of `run`: build agent + ControlPlane, optionally subscribe printer."""
        agent = LocalDockerAgent(host_id="host-local")
        cp = ControlPlane(agent)

        if not quiet:
            def on_event(event: Event) -> None:
                """Print each event using Rich markup."""
                line = _render_event(event)
                if line is not None:
                    console.print(line)
            cp.bus.subscribe(on_event)

        try:
            run = await cp.run_flow(flow)
        finally:
            await agent.shutdown()

        return 0 if run.state == FlowState.succeeded else 1

    rc = asyncio.run(_go())
    raise typer.Exit(rc)


if __name__ == "__main__":
    app()
