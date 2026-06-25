import asyncio

import pytest

from orchestration.control_plane import ControlPlane
from orchestration.monitoring import JsonlSink, MonitoringService
from orchestration.types import Flow, SubFlow

from test_nested_flows import FakeAgent, _Img, _training_subflow


def _build_nested_flow() -> Flow:
    """prep -> two training subflows (each train -> robustness) -> gather."""
    flow = Flow("sweep")
    prep = flow.add(_Img("prep"))
    subs = [flow.add(_training_subflow(t), after=prep) for t in ("a", "b")]
    flow.add(_Img("gather"), after=subs)
    return flow


def _run_with_monitor(sink=None) -> tuple[MonitoringService, str]:
    """Run the nested flow under a monitored control plane; return (monitor, run_id)."""
    agent = FakeAgent()
    cp = ControlPlane(agent)
    monitor = MonitoringService(sink=sink)
    cp.bus.subscribe(monitor.handle)
    run = asyncio.run(cp.run_flow(_build_nested_flow()))
    return monitor, run.id


def test_tree_reflects_nested_state() -> None:
    """The monitor reconstructs the parent run, its steps, and nested child runs."""
    monitor, root_id = _run_with_monitor()

    tree = monitor.tree(root_id)
    assert tree["flow"] == "sweep"
    assert tree["state"] == "succeeded"

    steps = {s["step"]: s for s in tree["steps"]}
    for tag in ("a", "b"):
        node = steps[f"hp-{tag}"]
        assert node["state"] == "succeeded"
        assert len(node["subflows"]) == 1
        child = node["subflows"][0]
        assert child["flow"] == f"train-{tag}"
        child_steps = {s["step"] for s in child["steps"]}
        assert child_steps == {f"train-{tag}", f"robustness-{tag}"}

    assert "subflows" not in steps["prep"]


def test_summary_counts_every_node() -> None:
    """summary() aggregates step states across the parent and all child runs."""
    monitor, _ = _run_with_monitor()
    summary = monitor.summary()
    assert summary.get("succeeded") == 8
    assert set(summary) == {"succeeded"}


def test_replay_reproduces_view(tmp_path) -> None:
    """Replaying the persisted JSONL log rebuilds an identical tree."""
    log = tmp_path / "events.jsonl"
    sink = JsonlSink(str(log))
    live, root_id = _run_with_monitor(sink=sink)
    sink.close()

    replayed = MonitoringService.replay_file(str(log))
    assert replayed.tree(root_id) == live.tree(root_id)
    assert replayed.summary() == live.summary()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
