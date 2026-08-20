"""Picking a session back up, whoever asked for it.

The terminal and the browser must make the same decision — a session already
running is focused rather than started again — so the decision lives here and
neither of them makes it.
"""

from dataclasses import dataclass
from pathlib import Path

from agent_sessions import attending, index, wormhole
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
    remote_home: str = ""


def resume(session: dict, fork: bool = False, at: str = "", remote: bool = False) -> Resumed:
    """Pick a session up, at its leaf or at a point inside it.

    Resuming at a point is always a fork: the session it came from is left as
    it is, and what is picked up is a new one that ends there.

    `remote` hands the session to the agent's own remote control, so it can be
    had from a phone. It is the one thing the agent cannot do for itself: its
    remote control reaches the session in front of you, and only the terminal
    can pick up an older one. That is what this index is for.
    """
    # First, because it is the one an agent cannot tell us about itself: a
    # headless turn registers nowhere, so `live()` below would not see it.
    if attending.held(session["native_id"]):
        raise NotResumable(
            f"{session['id']} has a turn already running on it from the control channel."
            " Resuming would put two agents on one transcript."
        )
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
        wormhole.run(
            project=session["project"],
            cwd=cwd,
            command=source.resume_command(native, remote=remote),
        )
        return Resumed(
            id=session["id"],
            project=session["project"],
            forked=True,
            was_running="",
            at=at,
            resumed_as=f"{session['agent']}:{native}",
            remote_home=source.remote_home if remote else "",
        )

    running = source.live().get(session["native_id"])
    # A session already running wants focusing, not starting again — unless
    # forking, which is a request for a second session beside the first.
    focusing = bool(running) and not fork
    wormhole.run(
        project=session["project"],
        cwd=cwd,
        # Nothing is started when a pane is being focused, so there is no
        # command line to put remote control on. Starting a second agent on the
        # same transcript would not do it either: it leaves remote control off
        # and the first one running.
        command=source.resume_command(
            session["native_id"], fork=fork, remote=remote and not focusing
        ),
        pid=running.pid if focusing else None,
    )
    return Resumed(
        id=session["id"],
        project=session["project"],
        forked=fork,
        was_running=running.status if running else "",
        remote_home=source.remote_home if remote and not focusing else "",
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
