"""Core user-facing types: Resources, Step, Flow."""
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Resources:
    """CPU and memory caps applied to each step's container."""

    cpu: float = 1.0
    memory_gb: float = 1.0


class Step:
    """Declarative unit of work: a Docker image + command + bind mounts."""

    image: str = ""
    command: list[str] | None = None
    env: dict[str, str] = {}
    resources: Resources = Resources()
    volumes: dict[str, str] = {}
    workdir: str | None = None

    def __init__(self, name: str | None = None):
        """Validate image is set and default `name` to the subclass name."""
        if not self.image:
            raise ValueError(f"{type(self).__name__}.image must be set")
        self.name = name or type(self).__name__


@dataclass
class _Node:
    """Internal DAG node: a Step plus the names of steps it depends on."""

    step: Step
    after: tuple[str, ...]


class Flow:
    """A named DAG of Steps assembled via `add(step, after=...)`."""

    def __init__(self, name: str):
        """Create an empty flow with the given name."""
        self.name = name
        self._nodes: dict[str, _Node] = {}

    def add(self, step: Step, after: Step | Iterable[Step] | None = None) -> Step:
        """Register `step` with optional upstream deps; returns the step for chaining."""
        if step.name in self._nodes:
            raise ValueError(f"duplicate step name: {step.name}")
        if after is None:
            deps: tuple[str, ...] = ()
        elif isinstance(after, Step):
            deps = (after.name,)
        else:
            deps = tuple(s.name for s in after)
        for d in deps:
            if d not in self._nodes:
                raise ValueError(f"step {step.name} depends on unknown step {d}")
        self._nodes[step.name] = _Node(step=step, after=deps)
        return step

    def steps(self) -> list[Step]:
        """Return all registered steps in insertion order."""
        return [n.step for n in self._nodes.values()]

    def deps(self, step_name: str) -> tuple[str, ...]:
        """Return the names of `step_name`'s immediate dependencies."""
        return self._nodes[step_name].after

    def topological_order(self) -> list[str]:
        """Kahn's algorithm. Raises if a cycle is present."""
        in_degree = {name: 0 for name in self._nodes}
        successors: dict[str, list[str]] = {name: [] for name in self._nodes}
        for name, node in self._nodes.items():
            for dep in node.after:
                in_degree[name] += 1
                successors[dep].append(name)

        ready = [n for n, d in in_degree.items() if d == 0]
        order: list[str] = []
        while ready:
            n = ready.pop(0)
            order.append(n)
            for s in successors[n]:
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    ready.append(s)
        if len(order) != len(self._nodes):
            raise ValueError("flow contains a cycle")
        return order
