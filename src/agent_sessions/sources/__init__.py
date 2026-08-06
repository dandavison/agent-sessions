"""Sources: one per agent whose transcripts we can read."""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from agent_sessions.models import Delta, Discovered, Running


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

    def resumable_from(self, path: Path, cwd: str) -> bool:
        """Whether an agent started in `cwd` would find this transcript.

        Each agent files its transcripts by the directory the session was had
        in, so the directory to resume in is not a free choice.
        """
        ...

    def fork_at(self, path: Path, at_uuid: str) -> str:
        """Write a session that ends at `at_uuid`, and return its native id.

        An agent resumes a session where it was left, so picking one up from
        earlier means writing the session it would have been. The original is
        not touched.
        """
        ...
