"""Forgetting a session: the transcript moved aside, and the index told.

Not deleted. A session worth forgetting is one that was never worth keeping —
a question asked and answered, a command run in the wrong window — and the cost
of being wrong about that should not be the work itself. So the transcript goes
to a graveyard under `~/.agent-sessions`, out of the agent's reach and out of
the index, and moving it back is all it would take.

A session cannot be forgotten while its agent still has the file open, so this
is something done after quitting: the id to hand it is the one Claude prints on
its way out.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from agent_sessions import db, index, query

GRAVEYARD = Path.home() / ".agent-sessions" / "forgotten"


class StillRunning(ValueError):
    """A session whose agent is still writing to it. Quit it first."""


@dataclass(frozen=True, slots=True)
class Forgotten:
    id: str
    title: str
    moved_to: str


def forget(conn: sqlite3.Connection, session_id: str) -> Forgotten:
    """Move a session's transcript out of the agent's reach, and drop the index rows."""
    agent, native_id = session_id.split(":", 1)
    source = index.SOURCES[agent]
    if native_id in source.live():
        raise StillRunning(f"{session_id} is still running. Quit it, then forget it.")

    session = query.get(conn, session_id)
    path = Path(session["path"]) if session else source.locate(native_id)
    moved = source.expunge(path, GRAVEYARD / agent) if path and path.exists() else None

    db.forget(conn, [session_id])
    conn.commit()
    return Forgotten(
        id=session_id,
        title=(session["title"] if session else "") or "",
        moved_to=str(moved) if moved else "",
    )
