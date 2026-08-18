"""Giving a session a title of my own.

A title I chose has to go where the agent keeps titles, or the next sync reads
the old one back over it. So this is a round trip through the transcript, not an
update to the index.
"""

from pathlib import Path
from typing import Any

import orjson
import pytest
from conftest import ai_title, assistant, last_prompt, text_block, user

from agent_sessions import db, index, query
from agent_sessions.sources import claude

NATIVE = "7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"
ID = f"claude:{NATIVE}"


@pytest.fixture
def transcript(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(index.SOURCES["claude"], "root", tmp_path / "projects")
    records: list[dict[str, Any]] = [
        user("u1", None, "why is conform relocating worktrees"),
        assistant("a1", "u1", [text_block("Because of submodules.")], "req_1"),
        ai_title("Investigate worktree relocation"),
        last_prompt("a1"),
    ]
    path = tmp_path / "projects" / "-Users-dan-src-wormhole" / f"{NATIVE}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in records))
    return path


def test_a_title_of_mine_beats_the_one_the_agent_wrote(transcript: Path) -> None:
    claude.retitle(transcript, "doing")
    delta = index.SOURCES["claude"].ingest(transcript)
    assert delta is not None
    assert delta.session.title == "doing"


def test_the_agent_reads_it_as_one_of_its_own(transcript: Path) -> None:
    """Same record it writes when the title is changed in its own UI."""
    claude.retitle(transcript, "doing")
    written = orjson.loads(transcript.read_bytes().splitlines()[-1])
    assert written == {"type": "custom-title", "customTitle": "doing", "sessionId": NATIVE}


def test_renaming_again_wins(transcript: Path) -> None:
    """Sidecar records are last-write-wins, so the newest is the title."""
    claude.retitle(transcript, "doing")
    claude.retitle(transcript, "reading")
    delta = index.SOURCES["claude"].ingest(transcript)
    assert delta is not None
    assert delta.session.title == "reading"


def test_the_index_shows_it_without_waiting_for_a_sync(tmp_path: Path, transcript: Path) -> None:
    conn = db.connect(tmp_path / "index.db")
    index.sync(conn, ["claude"])
    claude.retitle(transcript, "doing")
    db.set_title(conn, ID, "doing")
    conn.commit()
    session = query.get(conn, ID)
    assert session is not None
    assert session["title"] == "doing"
