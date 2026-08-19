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
from agent_sessions.models import Running

NATIVE = "7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"


def encoded(cwd: Path) -> str:
    """Claude's name for a cwd's transcript directory: every / and . becomes a dash."""
    return str(cwd).replace("/", "-").replace(".", "-")


@pytest.fixture
def resumes(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """What wormhole was asked to do, without asking it."""
    calls: list[dict] = []
    monkeypatch.setattr(wormhole, "run", lambda **kw: calls.append(kw))
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
    assert resumed.resumed_as.split(":")[1] in call["command"]


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


# --- what to run is ours to say, not wormhole's ----------------------------


def test_the_command_line_is_composed_here(session: dict, resumes: list[dict]) -> None:
    """Wormhole runs a command in a pane and knows nothing about agents."""
    resume.resume(session)
    assert resumes[0]["command"] == f"claude -r {NATIVE}"


def test_forking_is_spelled_in_the_command(session: dict, resumes: list[dict]) -> None:
    resume.resume(session, fork=True)
    assert resumes[0]["command"] == f"claude -r {NATIVE} --fork-session"


def test_picking_a_session_up_remotely_says_so_in_the_command(
    session: dict, resumes: list[dict]
) -> None:
    """The agent's own remote control is what puts a session on a phone, not us."""
    resume.resume(session, remote=True)
    assert resumes[0]["command"] == f"claude -r {NATIVE} --remote-control"


def test_a_remote_resume_says_where_to_go_next(session: dict, resumes: list[dict]) -> None:
    """Nothing to look at on this machine: the session is somewhere else now."""
    resumed = resume.resume(session, remote=True)
    assert resumed.remote_home == "https://claude.ai/code"


def test_a_local_resume_sends_you_nowhere(session: dict, resumes: list[dict]) -> None:
    resumed = resume.resume(session)
    assert resumed.remote_home == ""


def test_a_point_can_be_picked_up_remotely_too(session: dict, resumes: list[dict]) -> None:
    resumed = resume.resume(session, at="u1", remote=True)
    assert "--remote-control" in resumes[0]["command"]
    assert resumed.remote_home == "https://claude.ai/code"


def test_a_running_session_cannot_be_taken_over_remotely(
    session: dict, resumes: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second agent on the same transcript leaves remote control off and the first one running.

    So there is nowhere to send the phone, and saying that beats sending it to a
    session list that will not have the session in it.
    """
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {NATIVE: Running(pid=4242, status="waiting")}
    )
    resumed = resume.resume(session, remote=True)
    assert resumed.was_running == "waiting"
    assert resumed.remote_home == ""


def test_a_fork_is_not_told_about_the_session_it_came_from(
    session: dict, resumes: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wormhole focuses what a pid names; a fork is a second session, so it gets none."""
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {NATIVE: Running(pid=4242, status="waiting")}
    )
    resume.resume(session, fork=True)
    assert resumes[0]["pid"] is None


def test_a_running_session_is_focused_but_a_fork_of_it_is_not(
    session: dict, resumes: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Focusing is for picking a session up; forking asks for a second one beside it."""
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {NATIVE: Running(pid=4242, status="waiting")}
    )
    resume.resume(session)
    resume.resume(session, fork=True)
    assert [call["pid"] for call in resumes] == [4242, None]
