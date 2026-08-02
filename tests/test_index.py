import os
from pathlib import Path
from typing import Any

import orjson
import pytest
from conftest import assistant, last_prompt, text_block, user

from senderos import db, index, wormhole
from senderos.sources.claude import ClaudeSource

SESSION = "7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"
OTHER = "0f9442bd-bdae-41e7-91b5-cd6da55deb95"


@pytest.fixture
def projects(tmp_path: Path) -> Path:
    return tmp_path / "projects"


@pytest.fixture(autouse=True)
def one_worktree(monkeypatch: pytest.MonkeyPatch) -> None:
    """No live wormhole in tests; attribution is exercised through its real class."""
    monkeypatch.setattr(
        wormhole,
        "worktrees",
        lambda: [wormhole.Worktree(project_key="wormhole", working_tree="/Users/dan/src/wormhole")],
    )


@pytest.fixture(autouse=True)
def only_claude(projects: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(index.SOURCES, "claude", ClaudeSource(projects))


def write(projects: Path, records: list[dict[str, Any]], session: str = SESSION) -> Path:
    project = projects / "-Users-dan-src-wormhole"
    project.mkdir(parents=True, exist_ok=True)
    path = project / f"{session}.jsonl"
    path.write_bytes(b"".join(orjson.dumps({**r, "sessionId": session}) + b"\n" for r in records))
    return path


def conversation(text: str = "why is conform relocating worktrees") -> list[dict[str, Any]]:
    return [
        user("u1", None, text),
        assistant("a1", "u1", [text_block("Because it reconciles submodules.")], "req_1"),
        last_prompt("a1"),
    ]


def test_sync_indexes_a_transcript(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    conn = db.connect(tmp_path / "index.db")

    result = index.sync(conn)

    assert result.indexed == 1
    assert result.nodes == 2
    row = conn.execute("SELECT * FROM sendero").fetchone()
    assert row["id"] == f"claude:{SESSION}"
    assert row["n_user_turns"] == 1


def test_sync_attributes_to_a_wormhole_project(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn)
    assert conn.execute("SELECT project FROM sendero").fetchone()[0] == "wormhole"


def test_sync_leaves_unknown_paths_unattributed(projects: Path, tmp_path: Path) -> None:
    records = conversation()
    for r in records:
        if "cwd" in r:
            r["cwd"] = "/tmp/somewhere-else"
    write(projects, records)
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn)
    assert conn.execute("SELECT project FROM sendero").fetchone()[0] is None


def test_sync_is_idempotent(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    conn = db.connect(tmp_path / "index.db")

    first = index.sync(conn)
    second = index.sync(conn)

    assert first == second
    assert conn.execute("SELECT count(*) FROM sendero").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM node").fetchone()[0] == 2


def test_sync_picks_up_appended_records(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn)

    write(projects, [*conversation(), user("u2", "a1", "and now this"), last_prompt("u2")])
    index.sync(conn)

    assert conn.execute("SELECT n_user_turns FROM sendero").fetchone()[0] == 2


def test_sync_counts_sidecar_only_transcripts_as_skipped(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    write(projects, [last_prompt("nothing")], session=OTHER)
    conn = db.connect(tmp_path / "index.db")

    result = index.sync(conn)

    assert (result.indexed, result.skipped) == (1, 1)


def test_sync_forgets_a_deleted_transcript(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    path = write(projects, conversation("a second one"), session=OTHER)
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn)

    path.unlink()
    result = index.sync(conn)

    assert result.forgotten == 1
    assert conn.execute("SELECT count(*) FROM sendero").fetchone()[0] == 1
    assert conn.execute("SELECT count(*) FROM node").fetchone()[0] == 2


def test_sync_makes_text_searchable(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn)

    hits = conn.execute(
        "SELECT node.text FROM node_fts JOIN node ON node.rowid = node_fts.rowid"
        " WHERE node_fts MATCH ?",
        ("relocating",),
    ).fetchall()
    assert len(hits) == 1


def test_stale_reports_everything_before_the_first_sync(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    conn = db.connect(tmp_path / "index.db")
    assert index.stale(conn) == 1


def test_stale_is_zero_straight_after_a_sync(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn)
    assert index.stale(conn) == 0


def test_sidecar_only_transcripts_do_not_read_as_permanently_stale(
    projects: Path, tmp_path: Path
) -> None:
    """They are never indexed, so a per-file mtime check would flag them forever."""
    write(projects, conversation())
    write(projects, [last_prompt("nothing")], session=OTHER)
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn)
    assert index.stale(conn) == 0


def test_stale_notices_a_changed_transcript(projects: Path, tmp_path: Path) -> None:
    path = write(projects, conversation())
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn)

    synced_at = db.synced_at(conn)
    assert synced_at is not None
    os.utime(path, (synced_at + 10, synced_at + 10))

    assert index.stale(conn) == 1


def test_stale_notices_a_deleted_transcript(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    path = write(projects, conversation("a second one"), session=OTHER)
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn)

    path.unlink()

    assert index.stale(conn) == 1


def test_the_same_session_in_two_project_dirs_is_one_sendero(
    projects: Path, tmp_path: Path
) -> None:
    """Converting a Cursor conversation writes the same session into two dirs."""
    for encoded in ("-Users-dan-src-wormhole", "-Users-dan-src-wormhole-workspace"):
        directory = projects / encoded
        directory.mkdir(parents=True)
        (directory / f"{SESSION}.jsonl").write_bytes(
            b"".join(orjson.dumps(r) + b"\n" for r in conversation())
        )

    conn = db.connect(tmp_path / "index.db")
    result = index.sync(conn)

    assert (result.indexed, result.duplicated) == (1, 1)
    assert conn.execute("SELECT count(*) FROM sendero").fetchone()[0] == 1


def test_indexed_counts_senderos_not_files(projects: Path, tmp_path: Path) -> None:
    write(projects, conversation())
    write(projects, conversation("another"), session=OTHER)
    conn = db.connect(tmp_path / "index.db")

    result = index.sync(conn)

    assert result.indexed == conn.execute("SELECT count(*) FROM sendero").fetchone()[0] == 2
