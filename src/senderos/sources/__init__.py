"""Sources: one per agent whose transcripts we can read."""

from pathlib import Path
from typing import Protocol

from senderos.models import Delta, Discovered


class Source(Protocol):
    name: str

    def discover(self) -> list[Discovered]:
        """Every transcript this source knows about."""
        ...

    def ingest(self, path: Path) -> Delta | None:
        """Read one transcript. None when it holds no conversation."""
        ...
