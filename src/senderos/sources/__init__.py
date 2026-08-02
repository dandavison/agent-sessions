"""Sources: one per agent whose transcripts we can read."""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from senderos.models import Delta, Discovered, Running


class Source(Protocol):
    name: str

    def discover(self) -> list[Discovered]:
        """Every transcript this source knows about."""
        ...

    def ingest(self, path: Path) -> Delta | None:
        """Read one transcript. None when it holds no conversation."""
        ...

    def live(self) -> dict[str, Running]:
        """Native id -> process, for whatever this agent is running right now."""
        ...

    def render(self, path: Path, tools: bool, whole: bool) -> Iterator[str]:
        """The transcript as markdown, straight from the file.

        Not from the index, which holds no tool output: each agent knows how
        its own transcript should read.
        """
        ...
