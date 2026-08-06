"""The web UI, whose whole point is that a link resumes.

Pages are exercised through `web.handle`, which is where the routing lives; the
socket underneath it is plumbing and has nothing to decide.
"""

from pathlib import Path

import orjson
import pytest
from conftest import assistant, text_block, tool_result, tool_use, user

from agent_sessions import db, index, web, wormhole
from agent_sessions.models import Node, Running, Session

ID = "claude:7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"
NATIVE = ID.split(":")[1]
STRAY = "claude:00000000-0000-0000-0000-000000000000"


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One session in a project, one with nowhere to be resumed, one real transcript."""
    transcript = tmp_path / f"{NATIVE}.jsonl"
    records = [
        user("u1", None, "why is conform relocating worktrees"),
        assistant("a1", "u1", [tool_use("Bash")]),
        tool_result("t1", "a1"),
        assistant("a2", "t1", [text_block("Because of submodules.")]),
    ]
    transcript.write_bytes(b"\n".join(orjson.dumps(r) for r in records))

    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.write_session(
        conn,
        Session(
            id=ID,
            agent="claude",
            native_id=NATIVE,
            path=str(transcript),
            cwd="/Users/dan/src/wormhole",
            project="wormhole",
            title="why is conform relocating worktrees",
            started_at=1_784_990_000,
            ended_at=1_785_000_000,
            n_user_turns=1,
            context_tokens=1000,
        ),
    )
    db.write_session(
        conn,
        Session(
            id=STRAY,
            agent="claude",
            native_id=STRAY.split(":")[1],
            path=str(tmp_path / "gone.jsonl"),
            cwd="/tmp/scratch",
            title="a session with no project",
            ended_at=1_785_000_001,
        ),
    )
    db.write_nodes(
        conn,
        [
            Node(ID, "u1", None, 0, "user", 1, "why is conform relocating worktrees"),
            Node(ID, "a1", "u1", 1, "assistant", 2, "Because of submodules.", context_tokens=1000),
        ],
    )
    db.reindex_fts(conn)
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


@pytest.fixture
def resumes(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """What wormhole was asked to do, without asking it."""
    calls: list[dict] = []

    def record(project: str, session: str, fork: bool = False, pid: int | None = None) -> None:
        calls.append({"project": project, "session": session, "fork": fork, "pid": pid})

    monkeypatch.setattr(wormhole, "resume", record)
    monkeypatch.setattr(index.SOURCES["claude"], "live", dict)
    return calls


# --- a link is enough to resume --------------------------------------------


def test_every_session_listed_carries_a_resume_link(indexed: Path) -> None:
    body = web.handle("/").body
    assert f"/resume/{ID}" in body
    assert f"/resume/{ID}?fork=1" in body


def test_following_the_resume_link_resumes(indexed: Path, resumes: list[dict]) -> None:
    response = web.handle(f"/resume/{ID}")
    assert resumes == [{"project": "wormhole", "session": NATIVE, "fork": False, "pid": None}]
    assert response.status == 303
    assert response.location.startswith(f"/session/{ID}?")
    assert "Resumed" in response.location


def test_the_fork_link_forks(indexed: Path, resumes: list[dict]) -> None:
    web.handle(f"/resume/{ID}", "fork=1")
    assert resumes[0]["fork"] is True


def test_a_running_session_is_focused_rather_than_started_again(
    indexed: Path, resumes: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {NATIVE: Running(pid=4242, status="waiting")}
    )
    response = web.handle(f"/resume/{ID}")
    assert resumes[0]["pid"] == 4242
    assert "Already+running" in response.location


def test_a_session_without_a_project_says_why_it_cannot_resume(
    indexed: Path, resumes: list[dict]
) -> None:
    response = web.handle(f"/resume/{STRAY}")
    assert response.status == 400
    assert "/tmp/scratch" in response.body
    assert resumes == []


def test_wormhole_being_down_is_reported_rather_than_raised(
    indexed: Path, resumes: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise wormhole.WormholeUnavailable()

    monkeypatch.setattr(wormhole, "resume", refuse)
    response = web.handle(f"/resume/{ID}")
    assert response.status == 502
    assert "wormhole is not running" in response.body


def test_resuming_something_unknown_is_a_404(indexed: Path, resumes: list[dict]) -> None:
    assert web.handle("/resume/claude:nope").status == 404
    assert resumes == []


# --- a link is enough to resume at a point ---------------------------------


def test_a_link_can_name_the_point_to_pick_up_from(indexed: Path, resumes: list[dict]) -> None:
    response = web.handle(f"/resume/{ID}@a1")
    (call,) = resumes
    assert call["project"] == "wormhole"
    assert call["session"] not in (NATIVE, None)
    assert call["fork"] is False
    assert response.status == 303
    assert "a1" in response.location


def test_resuming_at_a_point_leaves_the_session_it_came_from_alone(
    indexed: Path, resumes: list[dict], tmp_path: Path
) -> None:
    transcript = tmp_path / f"{NATIVE}.jsonl"
    before = transcript.read_bytes()
    web.handle(f"/resume/{ID}@a1")
    assert transcript.read_bytes() == before


def test_a_point_no_one_can_find_is_a_404(indexed: Path, resumes: list[dict]) -> None:
    assert web.handle(f"/resume/{ID}@nope").status == 404
    assert resumes == []


def test_the_shape_offers_to_resume_at_each_point(indexed: Path) -> None:
    assert f"/resume/{ID}@a1" in web.handle(f"/session/{ID}").body


# --- finding things --------------------------------------------------------


def test_the_list_is_newest_first(indexed: Path) -> None:
    body = web.handle("/").body
    assert body.index("no project") < body.index("relocating worktrees")


def test_searching_narrows_the_list(indexed: Path) -> None:
    body = web.handle("/", "q=relocating").body
    assert "relocating worktrees" in body
    assert "no project" not in body


def test_a_search_shows_what_matched(indexed: Path) -> None:
    assert "class=snippet" in web.handle("/", "q=submodules").body


def test_a_search_matching_nothing_says_so(indexed: Path) -> None:
    assert "Nothing matched" in web.handle("/", "q=kangaroo").body


def test_filters_reach_the_index(indexed: Path) -> None:
    assert "relocating worktrees" in web.handle("/", "project=wormhole").body
    assert "relocating worktrees" not in web.handle("/", "project=temporal").body


def test_a_malformed_filter_is_reported_on_the_page(indexed: Path) -> None:
    response = web.handle("/", "since=2y")
    assert response.status == 400
    assert "2y" in response.body


def test_a_malformed_query_is_reported_on_the_page(indexed: Path) -> None:
    response = web.handle("/", "q=AND")
    assert response.status == 400
    assert "class=banner" in response.body


# --- one session -----------------------------------------------------------


def test_a_session_page_shows_its_turns_and_context(indexed: Path) -> None:
    body = web.handle(f"/session/{ID}").body
    assert "relocating worktrees" in body
    assert "1k" in body
    assert "wormhole" in body


def test_a_session_page_offers_the_transcript_and_a_resume(indexed: Path) -> None:
    body = web.handle(f"/session/{ID}").body
    assert f"/transcript/{ID}" in body
    assert f"/resume/{ID}" in body


def test_a_prefix_identifies_a_session(indexed: Path) -> None:
    assert web.handle("/session/7e90a7c6").status == 200


def test_an_unknown_session_is_a_404(indexed: Path) -> None:
    assert web.handle("/session/claude:nope").status == 404


def test_an_unknown_path_is_a_404(indexed: Path) -> None:
    assert web.handle("/nowhere").status == 404


def test_a_lost_branch_does_not_strike_out_what_came_after_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Struck-through text reaches the children of the element it is set on.

    Which would read as though branches that were taken had been abandoned, so
    the styling goes on the label rather than the item holding the subtree.
    """
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.write_session(
        conn,
        Session(
            id=ID,
            agent="claude",
            native_id=NATIVE,
            path="/t/a.jsonl",
            project="wormhole",
            title="rewound once",
            leaf_uuid="c1",
            ended_at=1_785_000_000,
        ),
    )
    db.write_nodes(
        conn,
        [
            Node(ID, "u1", None, 0, "user", 1, "start"),
            Node(ID, "a1", "u1", 1, "assistant", 2, ""),
            Node(ID, "b1", "a1", 2, "assistant", 3, ""),
            Node(ID, "b2", "b1", 3, "assistant", 4, ""),
            Node(ID, "b3", "b1", 4, "assistant", 5, ""),
            Node(ID, "c1", "a1", 5, "assistant", 6, ""),
        ],
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", path)

    body = web.handle(f"/session/{ID}").body
    lost = body.index("class='abandoned'")
    assert body.index("</span>", lost) < body.index("<ul>", lost)


# --- the transcript --------------------------------------------------------


def test_the_transcript_comes_from_the_file(indexed: Path) -> None:
    assert "Because of submodules." in web.handle(f"/transcript/{ID}").body


def test_tool_calls_are_off_until_asked_for(indexed: Path) -> None:
    assert "Bash" not in web.handle(f"/transcript/{ID}").body
    assert "Bash" in web.handle(f"/transcript/{ID}", "tools=1").body


def test_a_transcript_that_has_gone_is_reported(indexed: Path) -> None:
    assert web.handle(f"/transcript/{STRAY}").status == 404


# --- markup ----------------------------------------------------------------


def test_what_was_said_cannot_become_markup(indexed: Path) -> None:
    conn = db.connect()
    conn.execute("UPDATE session SET title = ? WHERE id = ?", ("<script>alert(1)</script>", ID))
    conn.commit()
    conn.close()
    for route in ("/", f"/session/{ID}"):
        body = web.handle(route).body
        assert "<script>alert(1)</script>" not in body
        assert "&lt;script&gt;" in body


def test_pages_are_html(indexed: Path) -> None:
    response = web.handle("/")
    assert response.content_type.startswith("text/html")
    assert response.body.startswith("<!doctype html>")
