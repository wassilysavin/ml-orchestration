import asyncio
from pathlib import Path
from typing import Awaitable, Callable, Iterable

Source = Callable[[], Iterable[str]]
OnFire = Callable[[str], Awaitable[None]]


def directory_source(path: str | Path, *, suffix: str | None = None) -> Source:
    """A source listing regular files in `path` (optionally filtered by suffix)."""
    base = Path(path)

    def source() -> list[str]:
        """Return current filenames in the directory, or [] if it doesn't exist."""
        if not base.exists():
            return []
        return [
            entry.name
            for entry in sorted(base.iterdir())
            if entry.is_file() and (suffix is None or entry.name.endswith(suffix))
        ]

    return source


class Trigger:
    """Fires `on_fire(key)` once per new key observed from `source`."""

    def __init__(
        self, source: Source, on_fire: OnFire, *, seen: Iterable[str] | None = None
    ) -> None:
        """Bind the source and callback; `seen` pre-marks keys to ignore."""
        self._source = source
        self._on_fire = on_fire
        self._seen: set[str] = set(seen or ())

    def baseline(self) -> None:
        """Mark every currently-present key as seen, so only later arrivals fire."""
        self._seen.update(self._source())

    async def poll_once(self) -> list[str]:
        """Fire for each new key once; return the keys fired this poll, in order."""
        fired: list[str] = []
        for key in self._source():
            if key not in self._seen:
                self._seen.add(key)
                await self._on_fire(key)
                fired.append(key)
        return fired

    async def watch(
        self, *, interval: float = 2.0, iterations: int | None = None
    ) -> None:
        """Poll forever (or `iterations` times), sleeping `interval` between polls."""
        count = 0
        while True:
            await self.poll_once()
            count += 1
            if iterations is not None and count >= iterations:
                return
            await asyncio.sleep(interval)
