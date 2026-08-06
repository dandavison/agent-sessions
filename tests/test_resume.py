"""The handoff to wormhole, which is the one thing this tool does rather than reports.

An agent finds a session by looking in the directory it was had in — Claude
keeps one transcript directory per cwd — so resuming in the wrong directory
opens a terminal in which the agent finds nothing and exits. That is not
visible from the index, and it is what these are about.
"""

from pathlib import Path
from typing import Any

import orjson
import pytest
from conftest import assistant, last_prompt, text_block, user

from agent_sessions import index, resume, wormhole

NATIVE = "7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"


def encoded(cwd: Path) -> str:
    """Claude's name for a cwd's transcript directory: every / and . becomes a dash."""
    return str(cwd).replace("/", "-").replace(".", "-")


@pytest.fixture
def resumes(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """What wormhole was asked to do, without asking it."""
    calls: list[dict] = []
    monkeypatch.setattr(wormhole, "resume", lambda **kw: calls.append(kw))
    monkeypatch.setattr(index.SOURCES["claude"], "live", dict)
    return calls


@pytest.fixture
def session(tmp_path: Path) -> dict:
    """A session had in a directory that is still there, indexed as wormhole sees it."""
    cwd = tmp_path / "src" / "wormhole"
    cwd.mkdir(parents=True)
    return _indexed(tmp_path, cwd)


def _indexed(tmp_path: Path, cwd: Path, at: Path | None = None) -> dict:
    records: list[dict[str, Any]] = [
        user("u1", None, "why is conform relocating worktrees"),
        assistant("a1", "u1", [text_block("Because of submodules.")], "req_1"),
        last_prompt("a1"),
    ]
    projects = at or (tmp_path / "projects" / encoded(cwd))
    projects.mkdir(parents=True, exist_ok=True)
    path = projects / f"{NATIVE}.jsonl"
    path.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in records))
    return {
        "id": f"claude:{NATIVE}",
        "agent": "claude",
        "native_id": NATIVE,
        "path": str(path),
        "cwd": str(cwd),
        "project": "wormhole",
    }


def test_wormhole_is_told_where_the_session_was_had(session: dict, resumes: list[dict]) -> None:
    resume.resume(session)
    (call,) = resumes
    assert call["cwd"] == session["cwd"]


def test_the_directory_handed_over_is_the_one_the_agent_will_look_in(
    session: dict, resumes: list[dict]
) -> None:
    """The invariant the whole thing rests on: resume there and the transcript is found."""
    resume.resume(session)
    (call,) = resumes
    assert encoded(Path(call["cwd"])) == Path(session["path"]).parent.name


def test_the_project_is_still_named_so_the_window_is_the_right_one(
    session: dict, resumes: list[dict]
) -> None:
    resume.resume(session)
    assert resumes[0]["project"] == "wormhole"


def test_resuming_at_a_point_lands_in_the_same_place(session: dict, resumes: list[dict]) -> None:
    """The session written for a point sits beside the one it came from."""
    resumed = resume.resume(session, at="u1")
    (call,) = resumes
    assert call["cwd"] == session["cwd"]
    assert encoded(Path(call["cwd"])) == Path(session["path"]).parent.name
    assert call["session"] == resumed.resumed_as.split(":")[1]


def test_a_session_whose_directory_is_gone_is_not_resumed(
    tmp_path: Path, resumes: list[dict]
) -> None:
    """A removed worktree is most of the corpus. Say so, rather than open a dead terminal."""
    session = _indexed(tmp_path, tmp_path / "worktrees" / "gone")
    with pytest.raises(resume.NotResumable):
        resume.resume(session)
    assert resumes == []


def test_a_session_the_agent_could_not_find_from_there_is_not_resumed(
    tmp_path: Path, resumes: list[dict]
) -> None:
    """The cwd exists but is not the one the transcript belongs to: the agent would find nothing."""
    cwd = tmp_path / "src" / "wormhole"
    cwd.mkdir(parents=True)
    session = _indexed(tmp_path, cwd, at=tmp_path / "projects" / "-somewhere-else")
    with pytest.raises(resume.NotResumable):
        resume.resume(session)
    assert resumes == []
