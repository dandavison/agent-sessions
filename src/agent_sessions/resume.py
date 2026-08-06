"""Picking a session back up, whoever asked for it.

The terminal and the browser must make the same decision — a session already
running is focused rather than started again — so the decision lives here and
neither of them makes it.
"""

from dataclasses import dataclass
from pathlib import Path

from agent_sessions import index, wormhole
from agent_sessions.sources import Source


class NotResumable(ValueError):
    """A session with nowhere to go: no project, or no directory to be resumed in."""


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
    path = _transcript(session)
    cwd = _cwd(session, source, path)
    if at:
        native = source.fork_at(path, at)
        wormhole.resume(project=session["project"], session=native, cwd=cwd)
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
        project=session["project"],
        session=session["native_id"],
        cwd=cwd,
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
        raise NotResumable(f"{path} is gone. Run `agent-sessions sync`.")
    return path


def _cwd(session: dict, source: Source, path: Path) -> str:
    """Where the agent has to be started for it to find this session.

    Not the project's working tree, which is all wormhole could work out on its
    own: an agent looks for a session in the directory it was had in, and half
    of mine were had somewhere else — a worktree since removed, or a
    subdirectory of one. Started anywhere else it finds nothing and exits,
    leaving a terminal that looks like a resume that did not happen.
    """
    cwd = session["cwd"] or ""
    if not Path(cwd).is_dir():
        raise NotResumable(
            f"{session['id']} was had in {cwd or 'a directory that was never recorded'},"
            " which is gone. There is nowhere to resume it."
        )
    if not source.resumable_from(path, cwd):
        raise NotResumable(
            f"{session['id']} is filed under {path.parent.name}, which is not where"
            f" {session['agent']} looks when it is started in {cwd}."
        )
    return cwd
