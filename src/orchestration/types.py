from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Union

Condition = Callable[[Mapping[str, Any]], bool]


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
    expected_exit_codes: frozenset[int] = frozenset({0})

    def __init__(self, name: str | None = None):
        """Validate image is set and default `name` to the subclass name."""
        if not self.image:
            raise ValueError(f"{type(self).__name__}.image must be set")
        self.name = name or type(self).__name__


class SubFlow:
    """A node that runs a whole child Flow instead of a single container."""

    def __init__(self, build: Callable[[], "Flow"], name: str):
        """Wrap a Flow factory under a node `name` used for deps and state."""
        self.build = build
        self.name = name


Work = Union[Step, SubFlow]


def exited_with(dep: "Work | str", *codes: int) -> Condition:
    """Condition: the dependency `dep` finished with one of `codes` as its exit code."""
    name = dep if isinstance(dep, str) else dep.name

    def predicate(steps: Mapping[str, Any]) -> bool:
        """Return True iff `dep`'s recorded exit code is in `codes`."""
        run = steps.get(name)
        return run is not None and run.exit_code in codes

    return predicate


@dataclass
class _Node:
    """Internal DAG node: work, the names of its deps, and an optional run condition."""

    work: Work
    after: tuple[str, ...]
    condition: Condition | None = None


class Flow:
    """A named DAG of Steps and SubFlows assembled via `add(work, after=...)`."""

    def __init__(self, name: str):
        """Create an empty flow with the given name."""
        self.name = name
        self._nodes: dict[str, _Node] = {}

    def add(
        self,
        work: Work,
        after: Work | Iterable[Work] | None = None,
        when: Condition | None = None,
    ) -> Work:
        """Register `work` with optional deps and a `when` condition; return it."""
        if work.name in self._nodes:
            raise ValueError(f"duplicate step name: {work.name}")
        if after is None:
            deps: tuple[str, ...] = ()
        elif isinstance(after, (Step, SubFlow)):
            deps = (after.name,)
        else:
            deps = tuple(s.name for s in after)
        for d in deps:
            if d not in self._nodes:
                raise ValueError(f"step {work.name} depends on unknown step {d}")
        if when is not None and not deps:
            raise ValueError(f"step {work.name} has a `when` condition but no deps")
        self._nodes[work.name] = _Node(work=work, after=deps, condition=when)
        return work

    def steps(self) -> list[Work]:
        """Return all registered nodes (Steps and SubFlows) in insertion order."""
        return [n.work for n in self._nodes.values()]

    def work(self, name: str) -> Work:
        """Return the unit of work registered under `name`."""
        return self._nodes[name].work

    def condition(self, name: str) -> Condition | None:
        """Return the `when` condition gating `name`, or None if unconditional."""
        return self._nodes[name].condition

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
