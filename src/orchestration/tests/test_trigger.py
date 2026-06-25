import asyncio

import pytest

from orchestration.trigger import Trigger, directory_source


def _recording_trigger(keys: list[str], *, seen=None):
    """Build a Trigger over a mutable key list; return (trigger, fired-list)."""
    fired: list[str] = []

    async def on_fire(key: str) -> None:
        fired.append(key)

    return Trigger(lambda: list(keys), on_fire, seen=seen), fired


def test_fires_once_per_new_key() -> None:
    """Each key fires exactly once, even across repeated polls."""
    keys = ["a.csv", "b.csv"]
    trigger, fired = _recording_trigger(keys)

    asyncio.run(trigger.poll_once())
    asyncio.run(trigger.poll_once())

    assert fired == ["a.csv", "b.csv"]


def test_new_arrivals_fire_on_later_poll() -> None:
    """A key that appears after the first poll fires on the next one."""
    keys = ["a.csv"]
    trigger, fired = _recording_trigger(keys)

    asyncio.run(trigger.poll_once())
    keys.append("c.csv")
    asyncio.run(trigger.poll_once())

    assert fired == ["a.csv", "c.csv"]


def test_baseline_suppresses_preexisting() -> None:
    """baseline() marks current keys seen, so only later arrivals fire."""
    keys = ["old.csv"]
    trigger, fired = _recording_trigger(keys)
    trigger.baseline()

    asyncio.run(trigger.poll_once())
    assert fired == []

    keys.append("new.csv")
    asyncio.run(trigger.poll_once())
    assert fired == ["new.csv"]


def test_watch_runs_a_bounded_number_of_polls() -> None:
    """watch(iterations=n) polls n times then returns, firing new keys seen."""
    keys = ["a.csv"]
    trigger, fired = _recording_trigger(keys)
    asyncio.run(trigger.watch(interval=0, iterations=3))
    assert fired == ["a.csv"]


def test_directory_source_lists_files(tmp_path) -> None:
    """directory_source returns filenames, honors a suffix filter, tolerates absence."""
    missing = directory_source(tmp_path / "nope")
    assert missing() == []

    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "b.txt").write_text("y")
    (tmp_path / "sub").mkdir()

    assert directory_source(tmp_path)() == ["a.csv", "b.txt"]
    assert directory_source(tmp_path, suffix=".csv")() == ["a.csv"]


def test_trigger_over_directory_fires_for_dropped_file(tmp_path) -> None:
    """End-to-end: baseline an empty dir, drop a file, the trigger fires for it."""
    fired: list[str] = []

    async def on_fire(key: str) -> None:
        fired.append(key)

    trigger = Trigger(directory_source(tmp_path, suffix=".csv"), on_fire)
    trigger.baseline()
    (tmp_path / "drugs_2026_06.csv").write_text("data")
    asyncio.run(trigger.poll_once())

    assert fired == ["drugs_2026_06.csv"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
