"""Picking a session back up, whoever asked for it.

The terminal and the browser must make the same decision — a session already
running is focused rather than started again — so the decision lives here and
neither of them makes it.
"""

from dataclasses import dataclass
from pathlib import Path

from agent_sessions import index, wormhole


class NotResumable(ValueError):
    """A session with nowhere to go: no project means no worktree to resume it in."""


@dataclass(frozen=True, slots=True)
class Resumed:
    id: str
    project: str
    forked: bool
    was_running: str
    at: str = ""
    resumed_as: str = ""


def resume(session: dict, fork: bool = False, at: str = "") -> Resumed:
    """Pick a session up, at its leaf or at a point inside it.

    Resuming at a point is always a fork: the session it came from is left as
    it is, and what is picked up is a new one that ends there.
    """
    if not session["project"]:
        raise NotResumable(
            f"{session['id']} has no project: its cwd was {session['cwd']}."
            " There is nowhere to resume it."
        )
    source = index.SOURCES[session["agent"]]
    if at:
        native = source.fork_at(_transcript(session), at)
        wormhole.resume(session["project"], native)
        return Resumed(
            id=session["id"],
            project=session["project"],
            forked=True,
            was_running="",
            at=at,
            resumed_as=f"{session['agent']}:{native}",
        )

    running = source.live().get(session["native_id"])
    wormhole.resume(
        session["project"],
        session["native_id"],
        fork=fork,
        pid=running.pid if running else None,
    )
    return Resumed(
        id=session["id"],
        project=session["project"],
        forked=fork,
        was_running=running.status if running else "",
    )


def _transcript(session: dict) -> Path:
    path = Path(session["path"])
    if not path.exists():
        raise NotResumable(f"{path} is gone, so there is no point in it to pick up.")
    return path
