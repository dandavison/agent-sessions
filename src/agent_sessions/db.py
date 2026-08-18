"""The index: a derived cache over the transcript files, safe to delete at any time."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from agent_sessions.models import Compaction, Edge, Node, Session

SCHEMA_VERSION = 2

DB_PATH = Path.home() / ".agent-sessions" / "index.db"

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE session (
  id             TEXT PRIMARY KEY,
  agent          TEXT NOT NULL,
  native_id      TEXT NOT NULL,
  path           TEXT NOT NULL,
  cwd            TEXT,
  project        TEXT,
  git_branch     TEXT,
  title          TEXT,
  model          TEXT,
  started_at     INTEGER,
  ended_at       INTEGER,
  n_user_turns   INTEGER NOT NULL DEFAULT 0,
  n_messages     INTEGER NOT NULL DEFAULT 0,
  context_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens  INTEGER NOT NULL DEFAULT 0,
  dropped_tokens INTEGER NOT NULL DEFAULT 0,
  leaf_uuid      TEXT,
  is_sidechain   INTEGER NOT NULL DEFAULT 0,
  session_kind   TEXT,
  file_mtime     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX session_ended_at ON session (ended_at DESC);
CREATE INDEX session_project ON session (project);
CREATE UNIQUE INDEX session_path ON session (path);

-- A uuid names a record, not a record of one session: a fork is written as a
-- copy of its parent's records, uuids and all, so the same uuid is a node of
-- every session that shares that history. Keyed by uuid alone, the session
-- indexed last would take those nodes from the others.
CREATE TABLE node (
  session_id      TEXT NOT NULL REFERENCES session (id) ON DELETE CASCADE,
  uuid            TEXT NOT NULL,
  parent_uuid     TEXT,
  seq             INTEGER NOT NULL,
  role            TEXT NOT NULL,
  ts              INTEGER,
  text            TEXT NOT NULL,
  request_id      TEXT,
  context_tokens  INTEGER NOT NULL DEFAULT 0,
  is_branch_point INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (session_id, uuid)
);
CREATE INDEX node_session ON node (session_id, seq);

CREATE TABLE session_edge (
  child   TEXT NOT NULL,
  parent  TEXT NOT NULL,
  kind    TEXT NOT NULL,
  at_uuid TEXT,
  PRIMARY KEY (child, parent, kind)
);
CREATE INDEX session_edge_parent ON session_edge (parent);

-- Shared with a fork for the same reason a node is, and keyed the same way.
CREATE TABLE compaction (
  session_id          TEXT NOT NULL REFERENCES session (id) ON DELETE CASCADE,
  uuid                TEXT NOT NULL,
  ts                  INTEGER,
  trigger             TEXT,
  pre_tokens          INTEGER,
  post_tokens         INTEGER,
  logical_parent_uuid TEXT,
  anchor_uuid         TEXT,
  preserved_count     INTEGER,
  PRIMARY KEY (session_id, uuid)
);
CREATE INDEX compaction_session ON compaction (session_id);

CREATE VIRTUAL TABLE node_fts USING fts5 (
  text,
  content='node',
  content_rowid='rowid',
  tokenize='porter unicode61'
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open the index, creating it if this is the first run.

    DB_PATH is read now rather than bound as a default, so it can be pointed
    somewhere else.
    """
    path = path if path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    if conn.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
        _rebuild(conn)
    return conn


def _rebuild(conn: sqlite3.Connection) -> None:
    """Drop and recreate. The index is derived, so a schema change is not a migration."""
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()


def indexed_paths(conn: sqlite3.Connection) -> set[str]:
    return {row["path"] for row in conn.execute("SELECT path FROM session")}


def synced_at(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT value FROM meta WHERE key = 'synced_at'").fetchone()
    return int(row["value"]) if row else None


def set_synced_at(conn: sqlite3.Connection, when: int) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('synced_at', ?)", (str(when),))


def write_session(conn: sqlite3.Connection, s: Session) -> None:
    columns = [f.name for f in Session.__dataclass_fields__.values()]
    placeholders = ", ".join(f":{c}" for c in columns)
    conn.execute(
        f"INSERT OR REPLACE INTO session ({', '.join(columns)}) VALUES ({placeholders})",
        {c: _adapt(getattr(s, c)) for c in columns},
    )


def write_nodes(conn: sqlite3.Connection, nodes: Iterable[Node]) -> None:
    rows = [
        (
            n.session_id,
            n.uuid,
            n.parent_uuid,
            n.seq,
            n.role,
            n.ts,
            n.text,
            n.request_id,
            n.context_tokens,
            int(n.is_branch_point),
        )
        for n in nodes
    ]
    if not rows:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO node (session_id, uuid, parent_uuid, seq, role, ts, text,"
        " request_id, context_tokens, is_branch_point) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def write_edges(conn: sqlite3.Connection, edges: Iterable[Edge]) -> None:
    rows = [(e.child, e.parent, e.kind, e.at_uuid) for e in edges]
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO session_edge (child, parent, kind, at_uuid) VALUES (?,?,?,?)",
            rows,
        )


def write_compactions(conn: sqlite3.Connection, compactions: Iterable[Compaction]) -> None:
    rows = [
        (
            c.session_id,
            c.uuid,
            c.ts,
            c.trigger,
            c.pre_tokens,
            c.post_tokens,
            c.logical_parent_uuid,
            c.anchor_uuid,
            c.preserved_count,
        )
        for c in compactions
    ]
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO compaction (session_id, uuid, ts, trigger, pre_tokens,"
            " post_tokens, logical_parent_uuid, anchor_uuid, preserved_count)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )


def set_title(conn: sqlite3.Connection, session_id: str, title: str) -> None:
    """What the next sync will read from the transcript anyway, visible now."""
    conn.execute("UPDATE session SET title = ? WHERE id = ?", (title, session_id))


def forget(conn: sqlite3.Connection, session_ids: Iterable[str]) -> int:
    """Drop agent-sessions whose transcripts have gone, and everything hanging off them."""
    ids = [(i,) for i in session_ids]
    if not ids:
        return 0
    conn.executemany("DELETE FROM node WHERE session_id = ?", ids)
    conn.executemany("DELETE FROM compaction WHERE session_id = ?", ids)
    conn.executemany(
        "DELETE FROM session_edge WHERE child = ? OR parent = ?", [(i, i) for (i,) in ids]
    )
    conn.executemany("DELETE FROM session WHERE id = ?", ids)
    return len(ids)


def reindex_fts(conn: sqlite3.Connection) -> None:
    """Silent nodes carry no text, so they sit in the index inert and match nothing."""
    conn.execute("INSERT INTO node_fts (node_fts) VALUES ('rebuild')")


def _adapt(value: object) -> object:
    return int(value) if isinstance(value, bool) else value
