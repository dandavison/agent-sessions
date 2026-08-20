"""The loop: notice a prompt, run the turn it asks for, post what came back.

This is the only part that runs an agent without me watching, so what it
refuses matters as much as what it does. It will not take a prompt up twice, it
will not touch a session that is open in a terminal, and it says so in the
thread rather than failing where nobody is looking.
"""

from dataclasses import dataclass, field

import pytest

from agent_sessions import attend, channel, index
from agent_sessions.models import Running


@dataclass
class FakeChannel:
    """A control channel that keeps its issues and comments in memory."""

    issues_: list[channel.Issue]
    comments_: dict[int, list[channel.Comment]] = field(default_factory=dict)
    posted: list[tuple[int, str]] = field(default_factory=list)
    taken: list[int] = field(default_factory=list)
    bodies: dict[int, str] = field(default_factory=dict)

    def issues(self) -> list[channel.Issue]:
        return self.issues_

    def comments(self, number: int) -> list[channel.Comment]:
        return self.comments_.get(number, [])

    def post(self, number: int, body: str) -> None:
        self.posted.append((number, body))

    def take_up(self, comment_id: int) -> None:
        self.taken.append(comment_id)

    def set_body(self, number: int, body: str) -> None:
        self.bodies[number] = body


SESSION = {
    "id": "claude:7e90",
    "agent": "claude",
    "native_id": "7e90",
    "cwd": "/tmp/x",
    "project": "wormhole",
    "path": "/tmp/x/7e90.jsonl",
}


def issue(session_id: str = "claude:7e90") -> channel.Issue:
    return channel.Issue(number=4, title="t", body=f"| session | {session_id} |", url="")


def prompt(id: int = 11, body: str = "try it with -x", taken: bool = False) -> channel.Comment:
    return channel.Comment(id=id, body=body, author="dandavison", taken_up=taken)


@pytest.fixture
def turns(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """What the agent was asked to do, without asking it."""
    ran: list[dict] = []

    def fake(session: dict, text: str, allowed: list[str]) -> list[dict]:
        ran.append({"session": session["id"], "prompt": text, "allowed": allowed})
        return [{"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}}]

    monkeypatch.setattr(attend, "run_turn", fake)
    monkeypatch.setattr(index.SOURCES["claude"], "live", dict)
    return ran


@pytest.fixture
def found(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(attend, "lookup", lambda conn, id: SESSION if id == "claude:7e90" else None)


# --- the loop ----------------------------------------------------------------


def test_a_comment_of_mine_is_a_prompt(turns: list[dict], found) -> None:
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert turns[0]["prompt"] == "try it with -x"


def test_the_answer_goes_back_to_the_issue_it_came_from(turns: list[dict], found) -> None:
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    number, body = c.posted[0]
    assert number == 4
    assert "Done." in body


def test_a_prompt_is_taken_up_before_it_is_run(turns: list[dict], found) -> None:
    """A turn takes minutes. Without the eyes first, it looks like nothing happened."""
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert c.taken == [11]


def test_what_it_wrote_is_not_read_back_as_a_prompt(turns: list[dict], found) -> None:
    """The loop that would never end."""
    ours = channel.Comment(id=12, body=f"{channel.MARKER}\nDone.", author="dandavison")
    c = FakeChannel([issue()], {4: [ours]})
    attend.once(c, conn=None)
    assert turns == []


def test_a_prompt_already_taken_up_is_left_alone(turns: list[dict], found) -> None:
    c = FakeChannel([issue()], {4: [prompt(taken=True)]})
    attend.once(c, conn=None)
    assert turns == []


def test_every_prompt_waiting_is_answered(turns: list[dict], found) -> None:
    c = FakeChannel([issue()], {4: [prompt(11, "first"), prompt(12, "second")]})
    attend.once(c, conn=None)
    assert [t["prompt"] for t in turns] == ["first", "second"]


def test_the_body_is_kept_up_to_date_with_what_i_have_said(turns: list[dict], found) -> None:
    """The one view the timeline cannot give, so it has to be maintained."""
    c = FakeChannel([issue()], {4: [prompt(11, "first")]})
    attend.once(c, conn=None)
    assert "first" in c.bodies[4]


# --- what it refuses ----------------------------------------------------------


def test_a_session_open_in_a_terminal_is_not_touched(
    turns: list[dict], found, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two agents on one transcript would corrupt it. Say so where I will read it."""
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {"7e90": Running(pid=42, status="busy")}
    )
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert turns == []
    assert "running" in c.posted[0][1].lower()


def test_an_issue_naming_no_session_is_said_to_be_wrong(turns: list[dict], found) -> None:
    c = FakeChannel([issue(session_id="claude:nope")], {4: [prompt()]})
    attend.once(c, conn=None)
    assert turns == []
    assert "claude:nope" in c.posted[0][1]


def test_an_issue_without_frontmatter_is_ignored_not_guessed_at(turns: list[dict], found) -> None:
    """A stray issue in the repo is not an invitation to run an agent."""
    c = FakeChannel(
        [channel.Issue(number=9, title="t", body="just a note", url="")], {9: [prompt()]}
    )
    attend.once(c, conn=None)
    assert turns == []
    assert c.posted == []


# --- what it is allowed to do -------------------------------------------------


def test_only_the_allowlisted_tools_are_offered(turns: list[dict], found) -> None:
    """I am in the park. Anything not on this list has to fail rather than proceed."""
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert turns[0]["allowed"] == attend.ALLOWED
    assert not [t for t in attend.ALLOWED if t.startswith("Bash(rm")]


def test_the_allowlist_does_not_include_a_bare_bash(turns: list[dict], found) -> None:
    """`Bash` with no pattern is every command there is."""
    assert "Bash" not in attend.ALLOWED
