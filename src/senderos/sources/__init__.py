"""Sources: one per agent whose transcripts we can read."""

from pathlib import Path
from typing import Protocol

from senderos.models import Delta, Discovered


class Source(Protocol):
    name: str

    def discover(self) -> list[Discovered]:
        """Every transcript this source knows about, with enough to detect change."""
        ...

    def ingest(self, path: Path, from_offset: int) -> Delta:
        """Parse from `from_offset` to EOF. Zero means the whole file."""
        ...
