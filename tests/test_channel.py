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


@pytest.fixture(autouse=True)
def _unminted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installation token is cached in the module, so tests would share one."""
    monkeypatch.setattr(channel, "_minted", ("", 0.0))


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


# --- a bot of its own, so a reply is not my own activity ----------------------


def test_a_comment_by_a_bot_is_not_a_prompt() -> None:
    """The marker says it for old comments; the author says it for every new one.

    Posting as myself is why nothing notified me: GitHub does not tell you about
    your own activity, so the answer landing was silent and I reloaded to check.
    """
    assert channel.Comment(id=12, body="Done.", author="agent-work[bot]").is_ours
    assert not channel.Comment(id=11, body=MINE_BODY, author="dandavison").is_ours


def test_the_marker_still_says_it_for_what_was_posted_before() -> None:
    """A repo full of comments posted as me must not become a queue of prompts."""
    assert channel.Comment(id=12, body=OURS_BODY, author="dandavison").is_ours


def app(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AGENT_WORK_APP_ID", "12345")
    monkeypatch.setenv("AGENT_WORK_APP_KEY", str(tmp_path / "key.pem"))
    (tmp_path / "key.pem").write_text("-----BEGIN RSA PRIVATE KEY-----\nnot a key\n")


def test_the_app_is_used_when_it_is_configured(monkeypatch, tmp_path) -> None:
    """Explicitly configured, not silently preferred: without it, `gh` is the token."""
    app(monkeypatch, tmp_path)
    monkeypatch.setattr(channel, "_jwt", lambda app_id, key: "signed.jwt")
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/app/installations":
            return httpx.Response(200, json=[{"id": 999}])
        if request.url.path == "/app/installations/999/access_tokens":
            return httpx.Response(
                201, json={"token": "ghs_installation", "expires_at": "2099-01-01T00:00:00Z"}
            )
        return httpx.Response(200, json=[])

    monkeypatch.setattr(
        channel,
        "_new_client",
        lambda: httpx.Client(
            transport=httpx.MockTransport(handle), base_url="https://api.github.com"
        ),
    )
    assert channel.token() == "ghs_installation"
    assert seen[0].headers["authorization"] == "Bearer signed.jwt"


def test_the_installation_token_is_not_minted_for_every_call(monkeypatch, tmp_path) -> None:
    """It lasts an hour; minting one per poll would be two extra calls every two seconds."""
    app(monkeypatch, tmp_path)
    monkeypatch.setattr(channel, "_jwt", lambda app_id, key: "signed.jwt")
    minted = {"n": 0}

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            minted["n"] += 1
            return httpx.Response(
                201, json={"token": "ghs_x", "expires_at": "2099-01-01T00:00:00Z"}
            )
        return httpx.Response(200, json=[{"id": 999}])

    monkeypatch.setattr(
        channel,
        "_new_client",
        lambda: httpx.Client(
            transport=httpx.MockTransport(handle), base_url="https://api.github.com"
        ),
    )
    channel.token()
    channel.token()
    assert minted["n"] == 1


def test_without_the_app_it_is_still_whatever_gh_is_signed_in_as(monkeypatch) -> None:
    monkeypatch.delenv("AGENT_WORK_APP_ID", raising=False)
    monkeypatch.setattr(channel, "_gh_token", lambda: "gho_mine")
    assert channel.token() == "gho_mine"


# --- one comment that becomes the answer --------------------------------------


def test_posting_says_which_comment_it_made() -> None:
    """Progress is an edit to that comment, so its id has to come back."""
    c = channel_over({"POST /repos/dandavison/agent-work/issues/4/comments": {"id": 13}})
    assert c.post(4, "Working…") == 13


def test_a_comment_can_be_rewritten() -> None:
    seen: list[httpx.Request] = []
    c = channel_over({"PATCH /repos/dandavison/agent-work/issues/comments/13": {"id": 13}}, seen)
    c.edit(13, "Done.")
    assert seen[-1].method == "PATCH"
    assert b"Done." in seen[-1].content


def test_a_progress_comment_is_ours_but_is_not_an_answer() -> None:
    """Left behind by a turn that was killed, it must not look like a reply."""
    running = channel.Comment(id=13, body=f"{channel.RUNNING}\nWorking…", author="a[bot]")
    assert running.is_ours
    assert running.is_progress
