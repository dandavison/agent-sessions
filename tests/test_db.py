from pathlib import Path

from senderos import db
from senderos.models import Compaction, Edge, Node, Sendero


def make_sendero(id: str = "claude:a", path: str = "/t/a.jsonl", **kw) -> Sendero:
    return Sendero(id=id, agent="claude", native_id=id.split(":")[1], path=path, **kw)


def test_connect_creates_schema(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"sendero", "node", "sendero_edge", "compaction", "node_fts"} <= tables
    assert conn.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION


def test_schema_version_bump_rebuilds(tmp_path: Path) -> None:
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.write_sendero(conn, make_sendero())
    conn.commit()
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    conn.close()

    conn = db.connect(path)
    assert conn.execute("SELECT count(*) FROM sendero").fetchone()[0] == 0


def test_write_sendero_is_idempotent(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_sendero(conn, make_sendero(title="first"))
    db.write_sendero(conn, make_sendero(title="second"))
    rows = conn.execute("SELECT title FROM sendero").fetchall()
    assert [r["title"] for r in rows] == ["second"]


def test_write_sendero_round_trips_bools(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_sendero(conn, make_sendero(is_sidechain=True))
    assert conn.execute("SELECT is_sidechain FROM sendero").fetchone()[0] == 1


def test_known_mtimes_reports_what_was_last_read(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_sendero(conn, make_sendero(file_mtime=5))
    db.write_sendero(conn, make_sendero(id="claude:b", path="/t/b.jsonl", file_mtime=9))
    assert db.known_mtimes(conn) == {"/t/a.jsonl": 5, "/t/b.jsonl": 9}


def test_forget_cascades(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_sendero(conn, make_sendero())
    db.write_sendero(conn, make_sendero(id="claude:b", path="/t/b.jsonl"))
    db.write_nodes(
        conn,
        [
            Node(
                sendero_id="claude:a",
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
                sendero_id="claude:a",
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
    assert conn.execute("SELECT count(*) FROM sendero_edge").fetchone()[0] == 0
    assert conn.execute("SELECT count(*) FROM sendero").fetchone()[0] == 1


def test_fts_finds_node_text(tmp_path: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    db.write_sendero(conn, make_sendero())
    db.write_nodes(
        conn,
        [
            Node(
                sendero_id="claude:a",
                uuid="u1",
                parent_uuid=None,
                seq=0,
                role="user",
                ts=1,
                text="why is conform relocating worktrees",
            ),
            Node(
                sendero_id="claude:a",
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
    db.write_sendero(conn, make_sendero())
    db.write_nodes(
        conn,
        [
            Node(
                sendero_id="claude:a",
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
    db.write_sendero(conn, make_sendero())
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
