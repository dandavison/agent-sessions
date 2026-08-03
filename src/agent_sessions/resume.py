"""Picking a session back up, whoever asked for it.

The terminal and the browser must make the same decision — a session already
running is focused rather than started again — so the decision lives here and
neither of them makes it.
"""

from dataclasses import dataclass

from agent_sessions import index, wormhole


class NotResumable(ValueError):
    """A session with nowhere to go: no project means no worktree to resume it in."""


@dataclass(frozen=True, slots=True)
class Resumed:
    id: str
    project: str
    forked: bool
    was_running: str


def resume(session: dict, fork: bool = False) -> Resumed:
    if not session["project"]:
        raise NotResumable(
            f"{session['id']} has no project: its cwd was {session['cwd']}."
            " There is nowhere to resume it."
        )
    running = index.SOURCES[session["agent"]].live().get(session["native_id"])
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
