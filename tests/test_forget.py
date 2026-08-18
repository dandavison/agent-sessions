"""Forgetting a session, which is the one thing here that takes something away.

Nothing is deleted: what forgetting means is out of the index, out of the
agent's reach, and out of the prompts it offers back at its input line. A
session still running is refused, so this is for after quitting one — named by
the id it printed on its way out, which no sync has seen yet.
"""

from pathlib import Path
from typing import Any

import orjson
import pytest
from conftest import assistant, last_prompt, text_block, user

from agent_sessions import cli, db, forget, index, query
from agent_sessions.models import Running
from agent_sessions.sources import claude

NATIVE = "7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"
ID = f"claude:{NATIVE}"
OTHER = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Everything this touches, pointed inside tmp_path."""
    monkeypatch.setattr(index.SOURCES["claude"], "root", tmp_path / "projects")
    monkeypatch.setattr(claude, "HISTORY", tmp_path / "history.jsonl")
    monkeypatch.setattr(forget, "GRAVEYARD", tmp_path / "forgotten")
    monkeypatch.setattr(index.SOURCES["claude"], "live", dict)
    return tmp_path


@pytest.fixture
def transcript(home: Path) -> Path:
    records: list[dict[str, Any]] = [
        user("u1", None, "why is conform relocating worktrees"),
        assistant("a1", "u1", [text_block("Because of submodules.")], "req_1"),
        last_prompt("a1"),
    ]
    path = home / "projects" / "-Users-dan-src-wormhole" / f"{NATIVE}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in records))
    return path


@pytest.fixture
def conn(home: Path, transcript: Path):
    connection = db.connect(home / "index.db")
    index.sync(connection, ["claude"])
    assert query.get(connection, ID) is not None
    return connection


@pytest.fixture
def history(home: Path) -> Path:
    path = home / "history.jsonl"
    path.write_bytes(
        b"".join(
            orjson.dumps({"display": text, "sessionId": sid}) + b"\n"
            for text, sid in [("mine", NATIVE), ("someone else's", OTHER), ("mine again", NATIVE)]
        )
    )
    return path


# --- what forgetting does --------------------------------------------------


def test_the_transcript_goes_where_the_agent_will_not_find_it(conn, transcript: Path) -> None:
    forgotten = forget.forget(conn, ID)
    assert not transcript.exists()
    assert Path(forgotten.moved_to).exists()


def test_the_transcript_is_kept_rather_than_deleted(conn, transcript: Path) -> None:
    """Irreversible is not what was asked for: expunged from the tool, not from disk."""
    before = transcript.read_bytes()
    forgotten = forget.forget(conn, ID)
    assert Path(forgotten.moved_to).read_bytes() == before


def test_the_index_no_longer_holds_it(conn) -> None:
    forget.forget(conn, ID)
    assert query.get(conn, ID) is None


def test_a_later_sync_does_not_bring_it_back(conn) -> None:
    forget.forget(conn, ID)
    index.sync(conn, ["claude"])
    assert query.get(conn, ID) is None


def test_the_prompts_go_too(conn, history: Path) -> None:
    """A session is not gone while its prompts are still offered at the input line."""
    forget.forget(conn, ID)
    kept = history.read_bytes().decode()
    assert "mine" not in kept
    assert "someone else's" in kept


def test_prompts_written_while_we_read_survive(conn, history: Path, monkeypatch) -> None:
    """Other sessions append to that file as this runs, and appends must not be lost."""
    original = claude._read_bytes

    def append_behind_our_back(path: Path) -> bytes:
        read = original(path)
        with path.open("ab") as f:
            f.write(orjson.dumps({"display": "concurrent", "sessionId": OTHER}) + b"\n")
        return read

    monkeypatch.setattr(claude, "_read_bytes", append_behind_our_back)
    forget.forget(conn, ID)
    assert "concurrent" in history.read_bytes().decode()


def test_what_the_session_left_beside_the_transcript_goes_with_it(
    conn, transcript: Path, home: Path
) -> None:
    sidecar = transcript.with_suffix("")
    (sidecar / "tool-results").mkdir(parents=True)
    (sidecar / "tool-results" / "out.txt").write_text("noise")
    forget.forget(conn, ID)
    assert not sidecar.exists()
    assert list((home / "forgotten").rglob("out.txt"))


def test_forgetting_something_never_indexed_still_moves_it(conn, transcript: Path) -> None:
    """Marked before the first sync, and the index is not the point of truth anyway."""
    db.forget(conn, [ID])
    conn.commit()
    forget.forget(conn, ID)
    assert not transcript.exists()


# --- a running session is not forgotten out from under itself --------------


def test_a_running_session_is_refused(conn, transcript: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {NATIVE: Running(pid=1, status="busy")}
    )
    with pytest.raises(forget.StillRunning):
        forget.forget(conn, ID)
    assert transcript.exists()


# --- from the command line, which is where all of this is typed ------------


@pytest.fixture
def run(conn, home: Path, monkeypatch, capsys: pytest.CaptureFixture[str]):
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", home / "index.db")

    def invoke(*args: str) -> tuple[int, str, str]:
        capsys.readouterr()
        code = cli.run(list(args))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return invoke


def test_forgetting_by_id(run, transcript: Path) -> None:
    code, out, _ = run("forget", ID)
    assert code == 0
    assert not transcript.exists()
    assert "forgotten" in out


def test_a_prefix_is_enough_here_too(run, transcript: Path) -> None:
    code, _, _ = run("forget", "7e90a7c6")
    assert code == 0
    assert not transcript.exists()


@pytest.mark.parametrize("named_as", [f"claude:{OTHER}", OTHER])
def test_the_id_claude_prints_as_it_exits_needs_no_sync_first(
    run, transcript: Path, named_as: str
) -> None:
    """`claude --resume <uuid>`, copied off the screen: a session ended a moment ago.

    Making that wait for a sync would be asking me to index a session in order
    to say I do not want it.
    """
    unindexed = transcript.with_name(f"{OTHER}.jsonl")
    unindexed.write_bytes(transcript.read_bytes())
    code, _, _ = run("forget", named_as)
    assert code == 0
    assert not unindexed.exists()


def test_renaming_from_the_command_line(run, transcript: Path) -> None:
    code, out, _ = run("rename", ID, "doing")
    assert code == 0
    assert "doing" in out
    assert b"custom-title" in transcript.read_bytes()
