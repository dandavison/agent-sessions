"""Bringing the index into line with the transcript files.

The files are the point of truth. Reading all of them takes about a second, so
there is no incremental path to get wrong: every sync reads everything and the
index is simply replaced.
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from agent_sessions import db, wormhole
from agent_sessions.sources import Source
from agent_sessions.sources.claude import ClaudeSource

SOURCES: dict[str, Source] = {"claude": ClaudeSource()}


@dataclass(frozen=True, slots=True)
class SyncResult:
    indexed: int
    skipped: int
    duplicated: int
    forgotten: int
    nodes: int


def sync(conn: sqlite3.Connection, agents: list[str] | None = None) -> SyncResult:
    """Read every transcript and rewrite the index from it."""
    attributor = wormhole.Attributor(wormhole.worktrees())
    sources = [SOURCES[a] for a in agents] if agents else list(SOURCES.values())

    seen: set[str] = set()
    skipped = duplicated = nodes = 0

    for source in sources:
        for found in source.discover():
            delta = source.ingest(found.path)
            if delta is None:
                skipped += 1
                continue
            if delta.session.id in seen:
                # The same session written into two project dirs, which wormhole
                # does when it converts a Cursor conversation. One session.
                duplicated += 1
                continue
            delta.session.project = attributor.project_for(delta.session.cwd)
            db.write_session(conn, delta.session)
            db.write_nodes(conn, delta.nodes)
            db.write_edges(conn, delta.edges)
            db.write_compactions(conn, delta.compactions)
            seen.add(delta.session.id)
            nodes += len(delta.nodes)

    forgotten = db.forget(conn, _vanished(conn, seen, sources))
    db.reindex_fts(conn)
    db.set_synced_at(conn, int(time.time()))
    conn.commit()
    return SyncResult(
        indexed=len(seen),
        skipped=skipped,
        duplicated=duplicated,
        forgotten=forgotten,
        nodes=nodes,
    )


def _vanished(conn: sqlite3.Connection, seen: set[str], sources: list[Source]) -> list[str]:
    """Indexed agent-sessions this sync did not find: their transcripts are gone."""
    agents = {s.name for s in sources}
    return [
        row["id"]
        for row in conn.execute("SELECT id, agent FROM session")
        if row["agent"] in agents and row["id"] not in seen
    ]


def stale(conn: sqlite3.Connection) -> int:
    """How many transcripts have changed since the last sync.

    A stat sweep, no writes: read commands report this so a stale index is
    visible rather than silently wrong. Measured against the time of the sync
    rather than per-file mtimes, so the handful of sidecar-only transcripts —
    which are never indexed and so have no stored mtime — do not read as
    permanently stale.
    """
    since = db.synced_at(conn)
    if since is None:
        return sum(len(source.discover()) for source in SOURCES.values())
    changed = sum(
        1 for source in SOURCES.values() for found in source.discover() if found.mtime > since
    )
    gone = sum(1 for path in db.indexed_paths(conn) if not Path(path).exists())
    return changed + gone
