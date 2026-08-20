"""The loop: notice a prompt, run the turn it asks for, post what came back.

This is the only part that runs an agent without me watching, so what it
refuses matters as much as what it does. It will not take a prompt up twice, it
will not touch a session that is open in a terminal, and it says so in the
thread rather than failing where nobody is looking.
"""

from dataclasses import dataclass, field

import pytest

from agent_sessions import attend, channel, index
from agent_sessions.models import Ran, Running, Said


@dataclass
class FakeChannel:
    """A control channel that keeps its issues and comments in memory."""

    issues_: list[channel.Issue]
    comments_: dict[int, list[channel.Comment]] = field(default_factory=dict)
    posted: list[tuple[int, str]] = field(default_factory=list)
    taken: list[int] = field(default_factory=list)
    bodies: dict[int, str] = field(default_factory=dict)
    titles: list[str] = field(default_factory=list)

    def open(self, title: str, body: str) -> channel.Issue:
        self.titles.append(title)
        made = channel.Issue(number=100 + len(self.issues_), title=title, body=body, url="u")
        self.issues_.append(made)
        return made

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

    def fake(session: dict, text: str) -> tuple[list[Said], dict]:
        ran.append({"session": session["id"], "prompt": text})
        return [Said(role="assistant", text="Done.")], {}

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


def test_a_session_open_in_a_terminal_is_taken_over(
    turns: list[dict], found, monkeypatch: pytest.MonkeyPatch
) -> None:
    """I am not at that keyboard. The pane loses; the prompt I just sent wins.

    Two agents on one transcript fork it and then fight over which branch is
    live, so one of them has to go, and it is not the one I am talking to.
    """
    killed: list[int] = []
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {"7e90": Running(pid=42, status="idle")}
    )
    monkeypatch.setattr(attend, "take_over", lambda pid: killed.append(pid))
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert killed == [42]
    assert turns[0]["prompt"] == "try it with -x"


def test_taking_a_session_over_is_said_in_the_thread(
    turns: list[dict], found, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Coming home to a closed pane should not be a mystery."""
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {"7e90": Running(pid=42, status="idle")}
    )
    monkeypatch.setattr(attend, "take_over", lambda pid: None)
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert "terminal" in c.posted[0][1].lower()


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


# --- opening the issue that makes a session reachable -------------------------


def test_a_session_gets_one_issue_not_one_per_visit(turns: list[dict]) -> None:
    """Asking twice is the normal case: the row is tapped whenever it is wanted."""
    c = FakeChannel([issue()])
    first = attend.issue_for(c, SESSION)
    second = attend.issue_for(c, SESSION)
    assert first.number == second.number == 4


def test_a_session_with_no_issue_yet_gets_one(turns: list[dict]) -> None:
    opened = FakeChannel([])
    made = attend.issue_for(opened, SESSION)
    assert made.session_id == "claude:7e90"


def test_the_new_issue_is_titled_as_the_session_is(turns: list[dict]) -> None:
    opened = FakeChannel([])
    attend.issue_for(opened, SESSION | {"title": "why is conform relocating"})
    assert opened.titles == ["why is conform relocating"]


# --- what it says about itself, and surviving a bad pass ---------------------


def test_a_pass_that_fails_does_not_end_the_loop(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A blip at GitHub while I am out must not be the end of it.

    The loop runs for hours unattended; an exception out of one pass is a
    reason to say so and poll again, not to stop answering until I get home.
    """

    class Broken(FakeChannel):
        def issues(self) -> list[channel.Issue]:
            raise RuntimeError("github said 502")

    attend.a_pass(Broken([]), conn=None)
    assert "github said 502" in capsys.readouterr().err


def test_noticing_a_prompt_is_logged(turns: list[dict], found, capsys) -> None:
    c = FakeChannel([issue()], {4: [prompt(11, "try it with -x")]})
    attend.once(c, conn=None)
    err = capsys.readouterr().err
    assert "#4" in err
    assert "try it with -x" in err


def test_how_long_a_turn_took_is_logged(turns: list[dict], found, capsys) -> None:
    """The question while waiting in a park is always `is it still going`."""
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert "s)" in capsys.readouterr().err


def test_taking_a_session_over_is_logged(
    turns: list[dict], found, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        index.SOURCES["claude"], "live", lambda: {"7e90": Running(pid=42, status="idle")}
    )
    monkeypatch.setattr(attend, "take_over", lambda pid: None)
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert "42" in capsys.readouterr().err


# --- saying something while a turn is running --------------------------------


def test_what_the_turn_does_is_said_as_it_does_it(capsys) -> None:
    """`--output-format json` says nothing until it exits, and a turn is minutes.

    The turn appends to the transcript as it goes, so the progress is already
    on disk; watching the file is how it reaches the terminal.
    """
    arriving = [
        [],
        [Ran(tool="Read", input={"file_path": "/x/a.py"})],
        [
            Ran(tool="Read", input={"file_path": "/x/a.py"}),
            Ran(tool="Bash", input={"command": "pytest"}),
        ],
    ]
    attend.watch(lambda: arriving.pop(0) if len(arriving) > 1 else arriving[0], alive=_alive(3))
    err = capsys.readouterr().err
    assert "Read" in err
    assert "pytest" in err


def test_a_block_is_not_reported_twice(capsys) -> None:
    same = [Ran(tool="Read", input={"file_path": "/x/a.py"})]
    attend.watch(lambda: same, alive=_alive(4))
    assert capsys.readouterr().err.count("Read") == 1


def test_a_turn_that_says_nothing_still_says_it_is_alive(capsys, monkeypatch) -> None:
    """Thinking for two minutes and a wedged process look identical otherwise."""
    monkeypatch.setattr(attend, "HEARTBEAT", 0.0)
    attend.watch(lambda: [], alive=_alive(2))
    assert "still going" in capsys.readouterr().err


def _alive(times: int):
    """A process that is running for `times` checks and then is not."""
    checks = iter([True] * times + [False] * 100)
    return lambda: next(checks)
