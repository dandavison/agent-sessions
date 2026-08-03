from pathlib import Path

from agent_sessions import db
from agent_sessions.models import Compaction, Edge, Node, Session


def make_session(id: str = "claude:a", path: str = "/t/a.jsonl", **kw) -> Session:
    return Session(id=id, agent="claude", native_id=id.split(":")[1], path=path, **kw)


def test_connect_creates_schema(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"session", "node", "session_edge", "compaction", "node_fts"} <= tables
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


def test_schema_version_bump_rebuilds(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.write_session(conn, make_session())
    conn.commit()
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()

    conn = db.connect(path)
    assert conn.execute("SELECT count(*) FROM session").fetchone()[0] == 0


def test_write_session_is_idempotent(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_session(conn, make_session(title="first"))
    db.write_session(conn, make_session(title="second"))
    rows = conn.execute("SELECT title FROM session").fetchall()
    assert [r["title"] for r in rows] == ["second"]


def test_write_session_round_trips_bools(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_session(conn, make_session(is_sidechain=True))
    assert conn.execute("SELECT is_sidechain FROM session").fetchone()[0] == 1


def test_forget_cascades(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_session(conn, make_session())
    db.write_session(conn, make_session(id="claude:b", path="/t/b.jsonl"))
    db.write_nodes(
        conn,
        [
            Node(
                session_id="claude:a",
                uuid="u1",
                parent_uuid=None,
                seq=0,
                role="user",
                ts=1,
                text="hi",
            )
        ],
    )
    db.write_edges(conn, [Edge(child="claude:b", parent="claude:a", kind="fork", at_uuid="u1")])
    db.write_compactions(
        conn,
        [
            Compaction(
                session_id="claude:a",
                uuid="c1",
                ts=2,
                trigger="auto",
                pre_tokens=100,
                post_tokens=10,
                logical_parent_uuid="u1",
                anchor_uuid="u2",
                preserved_count=3,
            )
        ],
    )

    assert db.forget(conn, ["claude:a"]) == 1
    assert conn.execute("SELECT count(*) FROM node").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM compaction").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM session_edge").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM session").fetchone()[0] == 1


def test_fts_finds_node_text(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_session(conn, make_session())
    db.write_nodes(
        conn,
        [
            Node(
                session_id="claude:a",
                uuid="u1",
                parent_uuid=None,
                seq=0,
                role="user",
                ts=1,
                text="why is conform relocating worktrees",
            ),
            Node(
                session_id="claude:a",
                uuid="u2",
                parent_uuid="u1",
                seq=1,
                role="user",
                ts=2,
                text="something entirely unrelated",
            ),
        ],
    )
    db.reindex_fts(conn)
    hits = conn.execute(
        "SELECT node.uuid FROM node_fts JOIN node ON node.rowid = node_fts.rowid"
        " WHERE node_fts MATCH ?",
        ("relocating",),
    ).fetchall()
    assert [h["uuid"] for h in hits] == ["u1"]


def test_fts_stems(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_session(conn, make_session())
    db.write_nodes(
        conn,
        [
            Node(
                session_id="claude:a",
                uuid="u1",
                parent_uuid=None,
                seq=0,
                role="user",
                ts=1,
                text="relocating worktrees",
            )
        ],
    )
    db.reindex_fts(conn)
    hits = conn.execute(
        "SELECT rowid FROM node_fts WHERE node_fts MATCH ?", ("relocate",)
    ).fetchall()
    assert len(hits) == 1


def test_silent_nodes_never_match_search(tmp_path: Path) -> None:
    """Tool calls are stored to keep the DAG whole, but carry no text to find."""
    conn = db.connect(tmp_path / "index.db")
    db.write_session(conn, make_session())
    db.write_nodes(
        conn,
        [
            Node("claude:a", "u1", None, 0, "user", 1, "find me"),
            Node("claude:a", "t1", "u1", 1, "assistant", 2, ""),
        ],
    )
    db.reindex_fts(conn)
    assert conn.execute("SELECT count(*) FROM node").fetchone()[0] == 2
    hits = conn.execute("SELECT rowid FROM node_fts WHERE node_fts MATCH ?", ("find",)).fetchall()
    assert len(hits) == 1


def test_synced_at_starts_unset_and_round_trips(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    assert db.synced_at(conn) is None
    db.set_synced_at(conn, 1785600000)
    assert db.synced_at(conn) == 1785600000


def test_indexed_paths_lists_every_transcript(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_session(conn, make_session())
    db.write_session(conn, make_session(id="claude:b", path="/t/b.jsonl"))
    assert db.indexed_paths(conn) == {"/t/a.jsonl", "/t/b.jsonl"}
