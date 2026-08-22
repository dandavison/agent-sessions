"""The loop against the real repo, because the doubles kept agreeing with me.

Every bug that reached the park got past a green suite: seven hundred comments
for seventy-three turns, a window that re-armed every prompt behind it, a token
bound once, markers matched by substring. None of them were in logic the unit
tests could reach — they were in what GitHub actually returns, and in whether
running the same pass twice does the same thing twice.

So this drives the real channel: a scratch issue, a synthetic transcript, and a
stubbed agent. The agent is stubbed because it is not what breaks; the channel
and the reconciler are.

    AGENT_WORK_LIVE=1 uv run pytest tests/test_live.py

Off by default. It creates and closes an issue in the control repo, and costs
real requests.
"""

import os
from pathlib import Path
from uuid import uuid4

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


@pytest.fixture
def live(scratch: dict):
    """A scratch issue in the real repo, closed again however the test ends."""
    control = channel.Channel()
    issue = attend.issue_for(control, scratch)
    try:
        yield control, issue
    finally:
        control._send("PATCH", f"/repos/{control.repo}/issues/{issue.number}", {"state": "closed"})


def test_a_prompt_is_answered_once_and_the_thread_settles(scratch, live, monkeypatch) -> None:
    """The whole loop, and then the same pass again changing nothing.

    Idempotence is the assertion that matters. Reconciling against one page of
    a longer thread reposted nearly every turn, for ever, and no unit test saw
    it because no double paginates.
    """
    control, issue = live
    monkeypatch.setattr(attend, "run_turn", _answers(scratch))

    mine = control.post(issue.number, "what is two and two?")
    attend.once(control, conn=None)

    after_one = control.comments(issue.number)
    renderings = [c for c in after_one if comment.turn_key(c.body)]
    assert len(renderings) == 1, "one turn, one rendering"
    assert "four" in renderings[0].body
    assert comment.asked_by(renderings[0].body) == mine

    settled = _fresh(control).comments(issue.number)
    attend.once(_fresh(control), conn=None)
    assert len(_fresh(control).comments(issue.number)) == len(settled), "a second pass adds nothing"


def test_a_turn_had_at_the_keyboard_shows_up_without_being_posted(scratch, live) -> None:
    """The guarantee the thread could not make before it was a projection."""
    control, issue = live
    _append(Path(scratch["path"]), _user("u9", "a0", "asked at the keyboard"))
    _append(Path(scratch["path"]), _assistant("a9", "u9", "answered at the keyboard"))

    attend.reconcile(control, _reread(control, issue), scratch)

    rendered = [c.body for c in _fresh(control).comments(issue.number) if comment.turn_key(c.body)]
    assert any("answered at the keyboard" in b for b in rendered)


def test_the_eyes_become_a_rocket(scratch, live, monkeypatch) -> None:
    """Eyes on means working, and that is only true if something takes them off."""
    control, issue = live
    monkeypatch.setattr(attend, "run_turn", _answers(scratch))
    mine = control.post(issue.number, "what is two and two?")
    attend.once(control, conn=None)

    reactions = control._get(f"/repos/{control.repo}/issues/comments/{mine}/reactions")
    kinds = {r["content"] for r in reactions}
    assert channel.DONE in kinds
    assert channel.SEEN not in kinds


def test_an_issue_starts_from_where_the_session_already_was(scratch, live) -> None:
    """Opening one is a way in, not a re-staging of everything said before."""
    _control, issue = live
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
    return next(i for i in _fresh(control).issues() if i.number == issue.number)


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
