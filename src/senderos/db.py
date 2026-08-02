"""The index: a derived cache over the transcript files, safe to delete at any time."""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from senderos.models import Compaction, Edge, Sendero, Turn

SCHEMA_VERSION = 1

DB_PATH = Path.home() / ".senderos" / "index.db"

SCHEMA = """
CREATE TABLE sendero (
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
  agent_type     TEXT,
  file_size      INTEGER NOT NULL DEFAULT 0,
  file_mtime     INTEGER NOT NULL DEFAULT 0,
  byte_offset    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX sendero_ended_at ON sendero (ended_at DESC);
CREATE INDEX sendero_project ON sendero (project);
CREATE UNIQUE INDEX sendero_path ON sendero (path);

CREATE TABLE turn (
  sendero_id     TEXT NOT NULL REFERENCES sendero (id) ON DELETE CASCADE,
  uuid           TEXT PRIMARY KEY,
  parent_uuid    TEXT,
  seq            INTEGER NOT NULL,
  role           TEXT NOT NULL,
  ts             INTEGER,
  text           TEXT NOT NULL,
  request_id     TEXT,
  context_tokens INTEGER NOT NULL DEFAULT 0,
  is_branch_point INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX turn_sendero ON turn (sendero_id, seq);

CREATE TABLE sendero_edge (
  child   TEXT NOT NULL,
  parent  TEXT NOT NULL,
  kind    TEXT NOT NULL,
  at_uuid TEXT,
  PRIMARY KEY (child, parent, kind)
);
CREATE INDEX sendero_edge_parent ON sendero_edge (parent);

CREATE TABLE compaction (
  sendero_id          TEXT NOT NULL REFERENCES sendero (id) ON DELETE CASCADE,
  uuid                TEXT PRIMARY KEY,
  ts                  INTEGER,
  trigger             TEXT,
  pre_tokens          INTEGER,
  post_tokens         INTEGER,
  logical_parent_uuid TEXT,
  anchor_uuid         TEXT,
  preserved_count     INTEGER
);
CREATE INDEX compaction_sendero ON compaction (sendero_id);

CREATE VIRTUAL TABLE turn_fts USING fts5 (
  text,
  content='turn',
  content_rowid='rowid',
  tokenize='porter unicode61'
);
"""


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    """Open the index, creating it if this is the first run."""
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


def cursors(conn: sqlite3.Connection, agent: str) -> dict[str, tuple[int, int, int]]:
    """Per-transcript ingest state: path -> (size, mtime, byte offset) as last read."""
    return {
        row["path"]: (row["file_size"], row["file_mtime"], row["byte_offset"])
        for row in conn.execute(
            "SELECT path, file_size, file_mtime, byte_offset FROM sendero WHERE agent = ?",
            (agent,),
        )
    }


def write_sendero(conn: sqlite3.Connection, s: Sendero) -> None:
    columns = [f.name for f in Sendero.__dataclass_fields__.values()]
    placeholders = ", ".join(f":{c}" for c in columns)
    conn.execute(
        f"INSERT OR REPLACE INTO sendero ({', '.join(columns)}) VALUES ({placeholders})",
        {c: _adapt(getattr(s, c)) for c in columns},
    )


def write_turns(conn: sqlite3.Connection, turns: Iterable[Turn]) -> None:
    rows = [
        (
            t.sendero_id,
            t.uuid,
            t.parent_uuid,
            t.seq,
            t.role,
            t.ts,
            t.text,
            t.request_id,
            t.context_tokens,
            int(t.is_branch_point),
        )
        for t in turns
    ]
    if not rows:
        return
    conn.executemany(
        "INSERT OR REPLACE INTO turn (sendero_id, uuid, parent_uuid, seq, role, ts, text,"
        " request_id, context_tokens, is_branch_point) VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )


def write_edges(conn: sqlite3.Connection, edges: Iterable[Edge]) -> None:
    rows = [(e.child, e.parent, e.kind, e.at_uuid) for e in edges]
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO sendero_edge (child, parent, kind, at_uuid) VALUES (?,?,?,?)",
            rows,
        )


def write_compactions(conn: sqlite3.Connection, compactions: Iterable[Compaction]) -> None:
    rows = [
        (
            c.sendero_id,
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
            "INSERT OR REPLACE INTO compaction (sendero_id, uuid, ts, trigger, pre_tokens,"
            " post_tokens, logical_parent_uuid, anchor_uuid, preserved_count)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            rows,
        )


def forget(conn: sqlite3.Connection, sendero_ids: Iterable[str]) -> int:
    """Drop senderos whose transcripts have gone, and everything hanging off them."""
    ids = [(i,) for i in sendero_ids]
    if not ids:
        return 0
    conn.executemany("DELETE FROM turn WHERE sendero_id = ?", ids)
    conn.executemany("DELETE FROM compaction WHERE sendero_id = ?", ids)
    conn.executemany(
        "DELETE FROM sendero_edge WHERE child = ? OR parent = ?", [(i, i) for (i,) in ids]
    )
    conn.executemany("DELETE FROM sendero WHERE id = ?", ids)
    return len(ids)


def reindex_fts(conn: sqlite3.Connection) -> None:
    conn.execute("INSERT INTO turn_fts (turn_fts) VALUES ('rebuild')")


def _adapt(value: object) -> object:
    return int(value) if isinstance(value, bool) else value
