"""The control channel: a private repo where issues are sessions and comments are turns.

Nothing listens on this machine. The laptop reaches out, which is the whole
reason for the design, so everything here is an outbound request and there is
no server to test.

Polling is how a comment is noticed, and a conditional request that has not
changed costs nothing against the rate limit — measured, not assumed — so the
ETag is not an optimisation to add later, it is what makes the interval a free
choice.
"""

import httpx
import pytest

from agent_sessions import channel


def responder(routes: dict[str, object], seen: list[httpx.Request] | None = None):
    """A GitHub that answers from a dict, and records what it was asked."""

    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        key = f"{request.method} {request.url.path}"
        found = routes.get(key)
        if found is None:
            return httpx.Response(404, json={"message": f"no route for {key}"})
        if isinstance(found, httpx.Response):
            return found
        return httpx.Response(200, json=found)

    return handle


def channel_over(routes: dict[str, object], seen: list[httpx.Request] | None = None):
    return channel.Channel(
        repo="dandavison/agent-work",
        client=httpx.Client(
            transport=httpx.MockTransport(responder(routes, seen)),
            base_url="https://api.github.com",
        ),
    )


ISSUE: dict[str, object] = {
    "number": 4,
    "title": "why is conform relocating",
    "body": "| session | claude:7e90 |",
}
MINE_BODY = "try it with -x"
OURS_BODY = "<!-- agent-work:turn -->\nDone."
MINE: dict[str, object] = {"id": 11, "body": MINE_BODY, "user": {"login": "dandavison"}}
OURS: dict[str, object] = {"id": 12, "body": OURS_BODY, "user": {"login": "dandavison"}}


# --- reading ----------------------------------------------------------------


def test_open_issues_are_the_sessions_being_worked_on() -> None:
    c = channel_over({"GET /repos/dandavison/agent-work/issues": [ISSUE]})
    (issue,) = c.issues()
    assert issue.number == 4
    assert issue.session_id == "claude:7e90"


def test_a_pull_request_is_not_an_issue() -> None:
    """GitHub returns PRs from the issues endpoint, and a PR is not a session."""
    pr = ISSUE | {"number": 5, "pull_request": {"url": "..."}}
    c = channel_over({"GET /repos/dandavison/agent-work/issues": [ISSUE, pr]})
    assert [i.number for i in c.issues()] == [4]


def test_comments_come_back_in_order() -> None:
    c = channel_over({"GET /repos/dandavison/agent-work/issues/4/comments": [MINE, OURS]})
    assert [x.id for x in c.comments(4)] == [11, 12]


# --- polling that costs nothing ---------------------------------------------


def test_the_etag_is_sent_back_on_the_next_poll() -> None:
    """Without this the poll interval is charged for; with it, it is free."""
    seen: list[httpx.Request] = []
    tagged = httpx.Response(200, json=[MINE], headers={"ETag": '"abc"'})
    c = channel_over({"GET /repos/dandavison/agent-work/issues/4/comments": tagged}, seen)
    c.comments(4)
    c.comments(4)
    assert seen[-1].headers.get("if-none-match") == '"abc"'


def test_nothing_new_means_nothing_changed_not_nothing_there() -> None:
    """A 304 has no body. Returning [] from it would look like the prompts vanished."""
    calls = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=[MINE], headers={"ETag": '"abc"'})
        return httpx.Response(304, headers={"ETag": '"abc"'})

    c = channel.Channel(
        repo="dandavison/agent-work",
        client=httpx.Client(
            transport=httpx.MockTransport(handle), base_url="https://api.github.com"
        ),
    )
    assert [x.id for x in c.comments(4)] == [11]
    assert [x.id for x in c.comments(4)] == [11]


# --- telling my comments from its own ----------------------------------------


def test_what_it_wrote_is_not_a_prompt() -> None:
    """It posts as me, so the author says nothing. The marker is what says it."""
    assert channel.Comment(id=12, body=OURS_BODY, author="dandavison").is_ours
    assert not channel.Comment(id=11, body=MINE_BODY, author="dandavison").is_ours


def test_a_prompt_already_taken_up_is_not_taken_up_again() -> None:
    """The eyes are the record that it was seen, and they live on GitHub.

    The index is a derived cache that is safe to delete; keeping this there
    would mean deleting it made the loop read its own output back as prompts.
    """
    reacted = MINE | {"reactions": {"eyes": 1}}
    c = channel_over({"GET /repos/dandavison/agent-work/issues/4/comments": [reacted]})
    assert c.comments(4)[0].taken_up


# --- writing -----------------------------------------------------------------


def test_a_posted_turn_carries_the_marker() -> None:
    seen: list[httpx.Request] = []
    c = channel_over({"POST /repos/dandavison/agent-work/issues/4/comments": {"id": 13}}, seen)
    c.post(4, "Done.")
    assert channel.MARKER in seen[-1].content.decode()


def test_taking_a_prompt_up_is_visible_from_the_park() -> None:
    """The eyes appear within seconds of posting, so the thing is seen to be alive."""
    seen: list[httpx.Request] = []
    c = channel_over(
        {"POST /repos/dandavison/agent-work/issues/comments/11/reactions": {"id": 1}}, seen
    )
    c.take_up(11)
    assert b"eyes" in seen[-1].content


def test_the_body_is_rewritten_in_place_not_appended_to() -> None:
    seen: list[httpx.Request] = []
    c = channel_over({"PATCH /repos/dandavison/agent-work/issues/4": ISSUE}, seen)
    c.set_body(4, "new body")
    assert seen[-1].method == "PATCH"


def test_opening_an_issue_returns_where_it_is() -> None:
    c = channel_over(
        {
            "POST /repos/dandavison/agent-work/issues": {
                "number": 7,
                "html_url": "https://github.com/dandavison/agent-work/issues/7",
                "body": "",
            }
        }
    )
    issue = c.open("why is conform relocating", "| session | claude:7e90 |")
    assert issue.number == 7
    assert issue.url.endswith("/issues/7")


# --- failing usefully ---------------------------------------------------------


def test_github_saying_no_is_not_swallowed() -> None:
    c = channel_over({})
    with pytest.raises(httpx.HTTPError):
        c.issues()


# --- and saying what it asked GitHub -----------------------------------------


def test_every_call_to_github_is_logged_in_detail(monkeypatch, capsys) -> None:
    """When nothing is happening, the question is whether it is even asking."""
    monkeypatch.setattr(channel.log, "VERBOSE", True)
    c = channel_over({"GET /repos/dandavison/agent-work/issues": [ISSUE]})
    c.issues()
    err = capsys.readouterr().err
    assert "/repos/dandavison/agent-work/issues" in err
    assert "200" in err
