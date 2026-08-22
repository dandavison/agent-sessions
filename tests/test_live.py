"""The loop against the real repo, because the doubles kept agreeing with me.

Every bug that reached the park got past a green suite: seven hundred comments
for seventy-three turns, a window that re-armed every prompt behind it, a token
bound once, markers matched by substring. None of them were in logic the unit
tests could reach — they were in what GitHub actually returns, and in whether
running the same pass twice does the same thing twice.

So this drives the real channel: a scratch issue, a synthetic transcript, and a
stubbed agent. The agent is stubbed because it is not what breaks; the channel
and the reconciler are.

Every pass names its issue. The first run of this reached into the two issues I
actually use and posted on both, which is the sort of thing a suite of doubles
will never tell you.

    AGENT_WORK_LIVE=1 uv run pytest tests/test_live.py

Off by default. It creates and closes an issue in the control repo, and costs
real requests.
"""

import os
from pathlib import Path
from uuid import uuid4

import httpx
import orjson
import pytest

from agent_sessions import attend, channel, comment, index
from agent_sessions.models import Said

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENT_WORK_LIVE") != "1", reason="AGENT_WORK_LIVE=1 to drive the real repo"
)


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A session of my own on disk, so the test owns everything it touches."""
    native = str(uuid4())
    path = tmp_path / f"{native}.jsonl"
    _append(path, _user("u0", None, "the first thing"))
    _append(path, _assistant("a0", "u0", "the first answer"))
    session = {
        "id": f"claude:{native}",
        "agent": "claude",
        "native_id": native,
        "path": str(path),
        "cwd": str(tmp_path),
        "project": "scratch",
        "title": f"live test {native[:8]}",
    }
    monkeypatch.setattr(attend, "lookup", lambda conn, id: session if id == session["id"] else None)
    return session


def _as_me() -> channel.Channel:
    """A channel authenticated as me rather than as the app.

    A prompt is a comment of mine. Posted through the app's channel it is
    authored by the bot, and the loop rightly ignores its own writing — so a
    test that posts that way is testing nothing, which is what the first three
    runs of this were doing.
    """
    return channel.Channel(
        client=httpx.Client(
            base_url=channel.API,
            headers={"Authorization": f"Bearer {channel._gh_token()}"},
            timeout=30.0,
        )
    )


@pytest.fixture
def live(scratch: dict):
    """A scratch issue in the real repo, closed again however the test ends."""
    control = channel.Channel()
    issue = attend.issue_for(control, scratch)
    try:
        yield control, issue, _as_me()
    finally:
        control._send("PATCH", f"/repos/{control.repo}/issues/{issue.number}", {"state": "closed"})


def test_a_prompt_is_answered_once_and_the_thread_settles(scratch, live, monkeypatch) -> None:
    """The whole loop, and then the same pass again changing nothing.

    Idempotence is the assertion that matters. Reconciling against one page of
    a longer thread reposted nearly every turn, for ever, and no unit test saw
    it because no double paginates.
    """
    control, issue, me = live
    monkeypatch.setattr(attend, "run_turn", _answers(scratch))

    mine = me.post(issue.number, "what is two and two?")
    attend.once(control, conn=None, only=issue.number)

    after_one = control.comments(issue.number)
    renderings = [c for c in after_one if comment.turn_key(c.body)]
    assert len(renderings) == 1, "one turn, one rendering"
    assert "four" in renderings[0].body
    assert comment.asked_by(renderings[0].body) == mine

    settled = _fresh(control).comments(issue.number)
    attend.once(_fresh(control), conn=None, only=issue.number)
    assert len(_fresh(control).comments(issue.number)) == len(settled), "a second pass adds nothing"


def test_a_turn_had_at_the_keyboard_shows_up_without_being_posted(scratch, live) -> None:
    """The guarantee the thread could not make before it was a projection."""
    control, issue, me = live
    _append(Path(scratch["path"]), _user("u9", "a0", "asked at the keyboard"))
    _append(Path(scratch["path"]), _assistant("a9", "u9", "answered at the keyboard"))

    attend.reconcile(control, _reread(control, issue), scratch)

    rendered = [c.body for c in _fresh(control).comments(issue.number) if comment.turn_key(c.body)]
    assert any("answered at the keyboard" in b for b in rendered)


def test_the_eyes_become_a_rocket(scratch, live, monkeypatch) -> None:
    """Eyes on means working, and that is only true if something takes them off."""
    control, issue, me = live
    monkeypatch.setattr(attend, "run_turn", _answers(scratch))
    mine = me.post(issue.number, "what is two and two?")
    attend.once(control, conn=None, only=issue.number)

    reactions = control._get(f"/repos/{control.repo}/issues/comments/{mine}/reactions")
    kinds = {r["content"] for r in reactions}
    assert channel.DONE in kinds
    assert channel.SEEN not in kinds


def test_an_issue_starts_from_where_the_session_already_was(scratch, live) -> None:
    """Opening one is a way in, not a re-staging of everything said before."""
    _control, issue, _me = live
    assert comment.frontmatter(issue.body)["from"]
    assert not [c for c in _fresh(_control).comments(issue.number) if comment.turn_key(c.body)]


# --- the scratch session ------------------------------------------------------


def _answers(session: dict):
    """An agent that writes a turn into the transcript, as the real one does."""

    def run_turn(_session, prompt, showing=None, tag=""):
        path = Path(session["path"])
        before = index.SOURCES["claude"].leaf(path)
        node = str(uuid4())
        _append(path, _user(node, before, prompt))
        _append(path, _assistant(str(uuid4()), node, "four"))
        return index.SOURCES["claude"].blocks(path, since=before), {"num_turns": 1}, before

    return run_turn


def _fresh(control: channel.Channel) -> channel.Channel:
    """A channel with no ETag cache, so an assertion reads GitHub and not memory."""
    return channel.Channel(repo=control.repo)


def _reread(control: channel.Channel, issue: channel.Issue) -> channel.Issue:
    """By name: the list does not have a new issue in it yet."""
    return _fresh(control).issue(issue.number)


def _append(path: Path, record: dict) -> None:
    with path.open("ab") as f:
        f.write(orjson.dumps(record) + b"\n")


def _user(uuid: str, parent: str | None, text: str) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "user", "content": text},
    }


def _assistant(uuid: str, parent: str, text: str) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


def _said(blocks) -> list[str]:
    return [b.text for b in blocks if isinstance(b, Said)]


def test_a_prompt_answered_before_a_compaction_is_not_run_again(scratch, live, monkeypatch) -> None:
    """The one that replayed nine answered prompts, against a real issue.

    A compaction writes a boundary whose `parentUuid` is null, so it starts a
    new root. Consumption was read by walking back from the leaf, which stops
    dead at that boundary — every prompt the session had already taken in fell
    out of view and re-armed. This session had answered forty-nine of them.
    """
    control, issue, me = live
    path = Path(scratch["path"])
    already = "what did we decide about the retry policy?"
    _append(path, _user("u5", "a0", already))
    _append(path, _assistant("a5", "u5", "we decided to keep it"))
    _append(
        path, {"type": "system", "subtype": "compact_boundary", "uuid": "cb", "parentUuid": None}
    )
    _append(path, _user("u6", "cb", "carry on"))
    _append(path, _assistant("a6", "u6", "carrying on"))

    ran: list[str] = []
    monkeypatch.setattr(attend, "run_turn", lambda *a, **k: ran.append(a[1]) or ([], {}, ""))
    me.post(issue.number, already)
    attend.once(control, conn=None, only=issue.number)

    assert ran == [], f"already answered before the compaction, but ran {ran}"
