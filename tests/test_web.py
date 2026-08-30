"""The web UI, whose whole point is that a link resumes.

Pages are exercised through `web.handle`, which is where the routing lives; the
socket underneath it is plumbing and has nothing to decide.
"""

import html
from pathlib import Path

import orjson
import pytest
from conftest import assistant, text_block, tool_result, tool_use, user

from agent_sessions import db, index, query, web, wormhole
from agent_sessions.models import Node, Running, Session
from agent_sessions.sources.claude import encode

ID = "claude:7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"
NATIVE = ID.split(":")[1]
STRAY = "claude:00000000-0000-0000-0000-000000000000"
ANSWER = "Because of **submodules**.\n\n```sh\ngit submodule status\n```"


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """One session in a project, one with nowhere to be resumed, one real transcript.

    The transcript sits where Claude files it — under the directory the session
    was had in — because resuming depends on the two agreeing.
    """
    cwd = tmp_path / "src" / "wormhole"
    cwd.mkdir(parents=True)
    projects = tmp_path / "projects" / encode(str(cwd))
    projects.mkdir(parents=True)
    transcript = projects / f"{NATIVE}.jsonl"
    records = [
        user("u1", None, "why is conform relocating worktrees"),
        assistant("a1", "u1", [tool_use("Bash")]),
        tool_result("t1", "a1"),
        assistant("a2", "t1", [text_block(ANSWER)]),
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
            cwd=str(cwd),
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

    monkeypatch.setattr(wormhole, "run", lambda **kw: calls.append(kw))
    monkeypatch.setattr(index.SOURCES["claude"], "live", dict)
    return calls


# --- a link is enough to resume --------------------------------------------


def test_every_session_listed_carries_a_resume_link(indexed: Path) -> None:
    assert f"/resume/{ID}" in web.handle("/").body


def test_following_the_resume_link_resumes(
    indexed: Path, resumes: list[dict], tmp_path: Path
) -> None:
    response = web.handle(f"/resume/{ID}")
    assert resumes[0] == {
        "project": "wormhole",
        "cwd": str(tmp_path / "src" / "wormhole"),
        "command": f"claude -r {NATIVE}",
        "pid": None,
    }
    assert response.status == 303
    assert response.location.startswith(f"/session/{ID}?")
    assert "Resumed" in response.location


def test_the_ui_does_not_fork(indexed: Path, resumes: list[dict]) -> None:
    """`/branch` in the session does this, and does it from wherever you have got to."""
    assert "fork" not in web.handle("/").body
    web.handle(f"/resume/{ID}", "fork=1")
    assert "--fork-session" not in resumes[0]["command"]


def test_a_running_session_is_focused_rather_than_started_again(
    indexed: Path, resumes: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {NATIVE: Running(pid=4242, status="waiting")}
    )
    response = web.handle(f"/resume/{ID}")
    assert resumes[0]["pid"] == 4242
    assert "Already+running" in response.location


# --- and a link is enough to pick it up on a phone -------------------------


def test_every_session_listed_can_be_put_on_the_control_channel(indexed: Path) -> None:
    """The reason to be reading this page on a phone at all.

    The rows used to offer the agent's own remote control. They offer the
    control channel instead — the remote-control route below still works, and
    the CLI still has `--remote`, but it needs a subscription this account does
    not have, and a row is not the place to find that out.
    """
    assert f"/issue/{ID}" in web.handle("/").body


def test_following_the_remote_link_hands_the_session_to_the_agents_own_remote_control(
    indexed: Path, resumes: list[dict]
) -> None:
    web.handle(f"/resume/{ID}", "remote=1")
    assert resumes[0]["command"] == f"claude -r {NATIVE} --remote-control"


def test_a_remote_resume_sends_the_browser_where_the_session_now_is(
    indexed: Path, resumes: list[dict]
) -> None:
    """Redirecting off this site is the point: the conversation is not here."""
    response = web.handle(f"/resume/{ID}", "remote=1")
    assert response.status == 303
    assert response.location == "https://claude.ai/code"


def test_a_running_session_cannot_be_picked_up_remotely(
    indexed: Path, resumes: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stay here and say so, rather than send a phone to a list without it in."""
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {NATIVE: Running(pid=4242, status="waiting")}
    )
    response = web.handle(f"/resume/{ID}", "remote=1")
    assert response.location.startswith(f"/session/{ID}?")
    assert "Already+running" in response.location


def test_a_point_can_be_picked_up_remotely(indexed: Path, resumes: list[dict]) -> None:
    response = web.handle(f"/resume/{ID}@a1", "remote=1")
    assert "--remote-control" in resumes[0]["command"]
    assert response.location == "https://claude.ai/code"


def test_actions_are_not_hidden_behind_a_hover_a_phone_cannot_do(indexed: Path) -> None:
    """Per-row actions fade in on hover, and a touch screen never hovers."""
    assert "hover: none" in web.CSS


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

    monkeypatch.setattr(wormhole, "run", refuse)
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
    assert NATIVE not in call["command"]
    assert "--fork-session" not in call["command"]
    assert response.status == 303
    assert "a1" in response.location


def test_resuming_at_a_point_leaves_the_session_it_came_from_alone(
    indexed: Path, resumes: list[dict], tmp_path: Path
) -> None:
    session = query.get(db.connect(), ID)
    assert session is not None
    transcript = Path(session["path"])
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


def test_a_session_page_offers_a_resume(indexed: Path) -> None:
    assert f"/resume/{ID}" in web.handle(f"/session/{ID}").body


def test_the_transcript_page_is_gone(indexed: Path) -> None:
    """One page shows a conversation now, and it is the session's own."""
    assert web.handle(f"/transcript/{ID}").status == 404


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


# --- ordering and paging ---------------------------------------------------


def test_a_column_heading_sorts_by_it(indexed: Path) -> None:
    body = web.handle("/").body
    assert "sort=turns" in body
    assert "sort=title" in body


def test_the_column_in_force_turns_around_when_clicked_again(indexed: Path) -> None:
    body = web.handle("/", "sort=turns").body
    assert "sort=turns&amp;reverse=1" in body
    assert "sort=context&amp;reverse=1" not in body


def test_reversing_reverses(indexed: Path) -> None:
    down = web.handle("/", "sort=title").body
    up = web.handle("/", "sort=title&reverse=1").body
    assert down.index("relocating worktrees") > down.index("no project")
    assert up.index("relocating worktrees") < up.index("no project")


def test_a_search_has_an_order_of_its_own(indexed: Path) -> None:
    """Results are ranked by how well they matched, so the headings do not sort."""
    assert "sort=turns" not in web.handle("/", "q=relocating").body


def test_a_filter_keeps_the_order_it_was_read_in(indexed: Path) -> None:
    assert "name=sort value='turns'" in web.handle("/", "sort=turns").body


def test_one_page_of_sessions_needs_no_pager(indexed: Path) -> None:
    assert "class=pager" not in web.handle("/").body


def test_a_full_page_offers_the_next(indexed: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "LIMIT", 1)
    body = web.handle("/").body
    assert "page=1" in body
    assert body.count("<tr>") == 2  # the headings, and one session


def test_the_last_page_offers_only_the_way_back(indexed: Path, monkeypatch) -> None:
    monkeypatch.setattr(web, "LIMIT", 1)
    body = web.handle("/", "page=1").body
    assert "page=2" not in body
    assert ">newer<" in body


def test_a_page_says_which_sessions_these_are(indexed: Path, monkeypatch) -> None:
    monkeypatch.setattr(web, "LIMIT", 1)
    assert "2–2" in web.handle("/", "page=1").body


def test_nonsense_paging_is_the_first_page(indexed: Path) -> None:
    assert web.handle("/", "page=-3").status == 200
    assert web.handle("/", "page=lots").status == 200


# --- filtering by project, as wormhole names them --------------------------


def test_the_project_half_of_a_task_filters_by_the_project(indexed: Path) -> None:
    """A task key is `project:branch`, and it is usually the project that is wanted."""
    assert "/?project=wormhole" in web.handle("/").body


def test_a_task_can_still_be_filtered_by(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.write_session(
        conn,
        Session(
            id="claude:t",
            agent="claude",
            native_id="t",
            path=str(tmp_path / "t.jsonl"),
            project="wormhole:dan/thing",
            title="a task of a project",
        ),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", path)

    body = web.handle("/").body
    assert "'/?project=wormhole'>wormhole</a>" in body
    assert "project=wormhole%3Adan%2Fthing" in body


def test_the_project_box_offers_what_there_is(indexed: Path) -> None:
    body = web.handle("/").body
    assert "<datalist id=projects>" in body
    assert "<option value='wormhole'>" in body


# --- and a link is enough to put it on the control channel -------------------


def test_following_the_issue_link_opens_one_and_goes_to_it(
    indexed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Leaving this site is the point: the conversation carries on over there."""
    opened: list[str] = []

    class Fake:
        def issues(self):
            return []

        def open(self, title: str, body: str):
            opened.append(title)
            return web.channel.Issue(
                number=7, title=title, body=body, url="https://github.com/x/y/issues/7"
            )

    monkeypatch.setattr(web, "control", Fake)
    response = web.handle(f"/issue/{ID}")
    assert response.status == 303
    assert response.location == "https://github.com/x/y/issues/7"
    assert opened == ["why is conform relocating worktrees"]


def test_a_session_that_already_has_an_issue_is_not_given_a_second(
    indexed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = web.channel.Issue(
        number=4, title="t", body=f"| session | {ID} |", url="https://github.com/x/y/issues/4"
    )

    class Fake:
        def issues(self):
            return [existing]

        def open(self, title: str, body: str):
            raise AssertionError("opened a second issue")

    monkeypatch.setattr(web, "control", Fake)
    assert web.handle(f"/issue/{ID}").location.endswith("/issues/4")


def test_github_being_unreachable_is_reported_not_raised(
    indexed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Fake:
        def issues(self):
            raise web.channel.NoToken()

    monkeypatch.setattr(web, "control", Fake)
    assert web.handle(f"/issue/{ID}").status == 502


# --- taking a turn away with you --------------------------------------------


def test_a_turn_carries_the_answer_it_got(indexed: Path) -> None:
    """The page held my prompts and nothing else, so there was nothing to take."""
    assert "submodules" in web.handle(f"/session/{ID}").body


def test_what_a_turn_ran_is_asked_for(indexed: Path) -> None:
    """Folded away is not the same as absent: tool output is most of the weight."""
    assert "Bash" not in web.handle(f"/session/{ID}").body
    assert "Bash" in web.handle(f"/session/{ID}", "tools=1").body


def test_each_turn_offers_itself_and_everything_after_it(indexed: Path) -> None:
    body = web.handle(f"/session/{ID}").body
    assert body.count("data-copy=turn") == 1
    assert body.count("data-copy=rest") == 1


def test_the_answer_is_folded_away_until_it_is_wanted(indexed: Path) -> None:
    body = web.handle(f"/session/{ID}").body
    assert "<details class=answer>" in body


def test_a_session_whose_transcript_has_gone_still_has_a_page(indexed: Path) -> None:
    """The index holds the turns; only what they carry needs the file."""
    response = web.handle(f"/session/{STRAY}")
    assert response.status == 200
    assert "data-copy" not in response.body


# --- a conversation, read as it was written ---------------------------------


def test_an_answer_is_rendered_not_recited(indexed: Path) -> None:
    """Markdown read as its own source is the one thing a browser is not needed for."""
    body = web.handle(f"/session/{ID}").body
    assert "<strong>submodules</strong>" in body
    assert "<code" in body
    assert "**submodules**" not in body.split("data-md=")[0]


def test_what_an_agent_wrote_cannot_become_markup(indexed: Path, tmp_path: Path) -> None:
    """Tool output is read off this machine and rendered into a page I then open."""
    conn = db.connect()
    found = query.get(conn, ID)
    conn.close()
    assert found is not None
    path = Path(found["path"])
    path.write_bytes(
        b"\n".join(
            orjson.dumps(r)
            for r in [
                user("u1", None, "what does it do"),
                assistant("a1", "u1", [text_block("<img src=x onerror=alert(1)>")]),
            ]
        )
    )
    body = web.handle(f"/session/{ID}").body
    assert "<img src=x" not in body
    assert "onerror" not in body or "&lt;img" in body


def test_the_markdown_is_still_there_to_be_copied(indexed: Path) -> None:
    """What is rendered cannot be pasted back; the source of it is what a copy takes."""
    body = web.handle(f"/session/{ID}").body
    assert "**submodules**" in html.unescape(body)


def test_the_buttons_are_icons_that_still_say_what_they_are(indexed: Path) -> None:
    body = web.handle(f"/session/{ID}").body
    assert "<svg" in body
    assert body.count("aria-label") >= 3


def test_a_turn_offers_to_be_resumed_from_where_the_row_does(indexed: Path) -> None:
    """The point stopped being the link; the action is the link, as in the index."""
    body = web.handle(f"/session/{ID}").body
    assert f"resume' href='/resume/{ID}@u1'" in body
    assert "class=point href=" not in body


def test_code_in_an_answer_is_highlighted(indexed: Path) -> None:
    """A fenced block says what language it is; rendering it grey throws that away."""
    body = web.handle(f"/session/{ID}").body
    assert "<span class=" in body.split("<details class=answer>")[1]
    assert ".hl" in body


def test_what_a_tool_was_given_is_highlighted_too(indexed: Path) -> None:
    """It is JSON, and it is the densest thing on the page when tools are on."""
    assert "hl" in web.handle(f"/session/{ID}", "tools=1").body.split("class=ran")[1]


def test_a_fence_in_no_language_is_still_shown(indexed: Path) -> None:
    """Nothing is guessed: an unlabelled block is code, and it is not decorated."""
    conn = db.connect()
    found = query.get(conn, ID)
    conn.close()
    assert found is not None
    Path(found["path"]).write_bytes(
        b"\n".join(
            orjson.dumps(r)
            for r in [
                user("u1", None, "show me"),
                assistant("a1", "u1", [text_block("```\nplain text\n```")]),
            ]
        )
    )
    assert "plain text" in web.handle(f"/session/{ID}").body
