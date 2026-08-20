"""Sources: one per agent whose transcripts we can read."""

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from agent_sessions.models import Block, Delta, Discovered, Running


class Source(Protocol):
    name: str

    # Where a session handed to this agent's remote control turns up, for the
    # browser that asked for it to be sent to. The agent's own surface: putting
    # a live conversation on a phone is its business, not this index's.
    remote_home: str

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

    def resume_command(self, native_id: str, fork: bool = False, remote: bool = False) -> str:
        """The command line that picks this session up, for a terminal to run.

        Each agent has its own, and nothing outside this package should have to
        know what it is: wormhole runs a command in a pane and has no opinion
        about which.

        `remote` asks for the session to be reachable from elsewhere. It still
        runs here — this is a command line for a pane on this machine — but the
        conversation is then had wherever the agent puts it.
        """
        ...

    def blocks(self, path: Path, since: str = "") -> list[Block]:
        """The conversation as blocks, which is what every reader wants.

        One parser per agent, and every place that shows a conversation is a
        formatter over its output. `since` gives what came after a point, which
        is how a turn just run is told from the rest of the session.
        """
        ...

    def leaf(self, path: Path) -> str:
        """Where the thread ends now, to ask what came after it later."""
        ...

    def turn_command(self, native_id: str, allowed: list[str]) -> list[str]:
        """Argv for one turn of this session with nobody at the keyboard.

        The prompt arrives on stdin. Nothing can be approved while it runs, so
        what it may do has to be settled here, before it starts.
        """
        ...

    def resumable_from(self, path: Path, cwd: str) -> bool:
        """Whether an agent started in `cwd` would find this transcript.

        Each agent files its transcripts by the directory the session was had
        in, so the directory to resume in is not a free choice.
        """
        ...

    def locate(self, native_id: str) -> Path | None:
        """This session's transcript, found without the index. None if it has gone."""
        ...

    def retitle(self, path: Path, title: str) -> None:
        """Give the session a title of my own, where the agent keeps its titles.

        Anywhere else and the next sync would read the old one back over it.
        """
        ...

    def expunge(self, path: Path, into: Path) -> Path:
        """Move this transcript, and everything else the session left, out of reach.

        Each agent keeps more than the transcript — prompts for recall at its
        input line, output too large to inline — and a session is not forgotten
        while any of it is still there.
        """
        ...

    def fork_at(self, path: Path, at_uuid: str) -> str:
        """Write a session that ends at `at_uuid`, and return its native id.

        An agent resumes a session where it was left, so picking one up from
        earlier means writing the session it would have been. The original is
        not touched.
        """
        ...
