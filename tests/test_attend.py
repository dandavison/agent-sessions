"""The loop: notice a prompt, run the turn it asks for, post what came back.

This is the only part that runs an agent without me watching, so what it
refuses matters as much as what it does. It will not take a prompt up twice, it
will not touch a session that is open in a terminal, and it says so in the
thread rather than failing where nobody is looking.
"""

from dataclasses import dataclass, field

import pytest

from agent_sessions import attend, channel, comment, index
from agent_sessions.models import Block, Ran, Running, Said


@dataclass
class FakeChannel:
    """A control channel that keeps its issues and comments in memory."""

    issues_: list[channel.Issue]
    comments_: dict[int, list[channel.Comment]] = field(default_factory=dict)
    posted: list[tuple[int, str]] = field(default_factory=list)
    edited: list[tuple[int, str]] = field(default_factory=list)
    taken: list[int] = field(default_factory=list)
    bodies: dict[int, str] = field(default_factory=dict)
    titles: list[str] = field(default_factory=list)
    deleted: list[int] = field(default_factory=list)
    done: list[int] = field(default_factory=list)

    def open(self, title: str, body: str) -> channel.Issue:
        self.titles.append(title)
        made = channel.Issue(number=100 + len(self.issues_), title=title, body=body, url="u")
        self.issues_.append(made)
        return made

    def issues(self) -> list[channel.Issue]:
        return self.issues_

    def comments(self, number: int) -> list[channel.Comment]:
        return self.comments_.get(number, [])

    def post(self, number: int, body: str) -> int:
        self.posted.append((number, body))
        return 900 + len(self.posted)

    def edit(self, comment_id: int, body: str) -> None:
        self.edited.append((comment_id, body))

    def delete(self, comment_id: int) -> None:
        self.deleted.append(comment_id)

    def take_up(self, comment_id: int) -> None:
        self.taken.append(comment_id)

    def mark_done(self, comment_id: int) -> None:
        self.done.append(comment_id)

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


def prompt(id: int = 11, body: str = "try it with -x") -> channel.Comment:
    return channel.Comment(id=id, body=body, author="dandavison")


@pytest.fixture
def turns(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """What the agent was asked to do, without asking it."""
    ran: list[dict] = []

    def fake(session: dict, text: str, showing=None, tag="") -> tuple[list[Said], dict, str]:
        ran.append({"session": session["id"], "prompt": text})
        return [Said(role="assistant", text="Done.")], {}, "leafbefore"

    monkeypatch.setattr(attend, "run_turn", fake)
    monkeypatch.setattr(index.SOURCES["claude"], "live", dict)
    # The body is read off the transcript now, and these sessions have none.
    monkeypatch.setattr(index.SOURCES["claude"], "blocks", lambda path, since="": [])
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
    """It lands in the comment posted when the turn began, not a new one."""
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert c.posted[0][0] == 4
    assert "Done." in c.edited[-1][1]


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


def test_the_eyes_alone_do_not_mean_a_prompt_is_done(turns: list[dict], found) -> None:
    """This asserted the opposite, and the opposite was the bug.

    Eyes with no reply is what a turn killed part way leaves behind, and
    treating it as done is what dropped that work. A reply is what settles it.
    """
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert turns[0]["prompt"] == "try it with -x"


def test_every_prompt_waiting_is_answered(turns: list[dict], found) -> None:
    c = FakeChannel([issue()], {4: [prompt(11, "first"), prompt(12, "second")]})
    attend.once(c, conn=None)
    assert [t["prompt"] for t in turns] == ["first", "second"]


def test_the_body_is_kept_up_to_date(turns: list[dict], found) -> None:
    """The one view the timeline cannot give, so it has to be maintained.

    It used to assert that a posted comment appears here; it asserts the
    frontmatter now, because the list beneath comes from the transcript rather
    than from the thread.
    """
    c = FakeChannel([issue()], {4: [prompt(11, "first")]})
    attend.once(c, conn=None)
    assert "claude:7e90" in c.bodies[4]


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
    assert "terminal" in c.edited[-1][1].lower()


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
    assert "answered in" in capsys.readouterr().err


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


@pytest.fixture(autouse=True)
def _brisk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The watcher's poll is a real second in production and none in a test."""
    monkeypatch.setattr(attend, "POLL", 0.0)


def test_what_the_turn_does_is_said_as_it_does_it(capsys) -> None:
    """`--output-format json` says nothing until it exits, and a turn is minutes.

    The turn appends to the transcript as it goes, so the progress is already
    on disk; watching the file is how it reaches the terminal.
    """
    read = Ran(tool="Read", input={"file_path": "/x/a.py"})
    tests = Ran(tool="Bash", input={"command": "pytest"})
    arriving: list[list[Block]] = [[], [read], [read, tests]]
    attend.watch(lambda: arriving.pop(0) if len(arriving) > 1 else arriving[0], alive=_alive(3))
    err = capsys.readouterr().err
    assert "Read" in err
    assert "pytest" in err


def test_a_block_is_not_reported_twice(capsys) -> None:
    same: list[Block] = [Ran(tool="Read", input={"file_path": "/x/a.py"})]
    attend.watch(lambda: same, alive=_alive(4))
    assert capsys.readouterr().err.count("Read") == 1


def test_a_turn_that_says_nothing_still_says_it_is_alive(capsys, monkeypatch) -> None:
    """Thinking for two minutes and a wedged process look identical otherwise."""
    monkeypatch.setattr(attend, "HEARTBEAT", 0.0)
    nothing: list[Block] = []
    attend.watch(lambda: nothing, alive=_alive(2))
    assert "still going" in capsys.readouterr().err


def _alive(times: int):
    """A process that is running for `times` checks and then is not."""
    checks = iter([True] * times + [False] * 100)
    return lambda: next(checks)


def test_it_says_where_the_turn_is_running(turns: list[dict], found, capsys) -> None:
    """Which worktree it landed in is the first thing I check when output surprises me."""
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert SESSION["cwd"] in capsys.readouterr().err


def test_it_says_that_it_posted_and_how_much(turns: list[dict], found, capsys) -> None:
    """A posted comment that never appears is a different bug from one never written."""
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert "posted" in capsys.readouterr().err


def test_it_says_how_many_prompts_are_waiting(turns: list[dict], found, capsys) -> None:
    c = FakeChannel([issue()], {4: [prompt(11, "first"), prompt(12, "second")]})
    attend.once(c, conn=None)
    assert "2 waiting" in capsys.readouterr().err


# --- a turn cut off part way is not finished ---------------------------------


def test_a_prompt_whose_turn_never_finished_is_picked_up_again(turns: list[dict], found) -> None:
    """Killing the loop mid-turn dropped the prompt for good.

    The eyes go on before the turn runs, so they say `started`, not `answered`.
    With them as the record, a turn interrupted by a restart was never retried
    and never replied to — the work was in the transcript and nowhere else.
    """
    interrupted = prompt(11, "the one that was cut off")
    c = FakeChannel([issue()], {4: [interrupted]})
    attend.once(c, conn=None)
    assert turns[0]["prompt"] == "the one that was cut off"


def test_a_prompt_that_was_answered_is_left_alone(turns: list[dict], found, monkeypatch) -> None:
    """Settled by the session having taken it in, not by a reply sitting near it."""
    transcript(monkeypatch, said_by_me("already done", "u1"), Said("assistant", "Done."))
    c = FakeChannel([issue()], {4: [prompt(11, "already done")]})
    attend.once(c, conn=None)
    assert turns == []


# --- progress where I am actually looking ------------------------------------


def test_a_comment_appears_before_the_turn_starts(turns: list[dict], found) -> None:
    """Posted from the park, the thread should show something before the answer."""
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert c.posted[0][1].startswith(channel.RUNNING)


def test_the_answer_replaces_that_comment_rather_than_adding_one(turns: list[dict], found) -> None:
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert len(c.posted) == 1
    assert "Done." in c.edited[-1][1]
    assert channel.RUNNING not in c.edited[-1][1]


def test_the_finished_comment_carries_the_marker(turns: list[dict], found) -> None:
    """Nothing else keeps the loop finite for comments the bot did not author."""
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert c.edited[-1][1].startswith(channel.MARKER)


def test_a_stale_progress_comment_does_not_count_as_an_answer(turns: list[dict], found) -> None:
    """What a killed turn leaves: the prompt is still unanswered and gets retried."""
    stale = channel.Comment(id=13, body=f"{channel.RUNNING}\nWorking…", author="a[bot]")
    c = FakeChannel([issue()], {4: [prompt(11, "cut off"), stale]})
    attend.once(c, conn=None)
    assert turns[0]["prompt"] == "cut off"


# --- the body is what the transcript says, not what the thread does ----------


def test_the_body_lists_the_turns_in_the_transcript(
    turns: list[dict], found, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt that never ran is on GitHub and not in the session.

    Listing comments made the body assert turns the agent has no knowledge of.
    The transcript is what it will resume from, so that is what the body shows.
    """
    monkeypatch.setattr(
        index.SOURCES["claude"],
        "blocks",
        lambda path, since="": [
            Said(role="user", text="what actually ran"),
            Said(role="assistant", text="an answer"),
        ],
    )
    c = FakeChannel([issue()], {4: [prompt(11, "never ran")]})
    attend.once(c, conn=None)
    assert "what actually ran" in c.bodies[4]
    assert "an answer" not in c.bodies[4]


# --- rewinding by tapping once ------------------------------------------------


def rewound(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    forks: list[str] = []

    def fork(path, at_uuid: str) -> str:
        forks.append(at_uuid)
        return "newnative"

    monkeypatch.setattr(index.SOURCES["claude"], "fork_at", fork)
    monkeypatch.setattr(attend, "reindex", lambda conn: None)
    return forks


def answered(at_uuid: str = "9f3c1d20", id: int = 12) -> channel.Comment:
    return channel.Comment(
        id=id, body=f"{channel.MARKER}\n{comment.at(at_uuid)}\nDone.", author="a[bot]"
    )


def test_a_thumbs_down_on_an_answer_forks_at_the_point_it_recorded(
    turns: list[dict], found, monkeypatch: pytest.MonkeyPatch
) -> None:
    forks = rewound(monkeypatch)
    down = channel.Comment(id=12, body=answered().body, author="a[bot]", rewind_wanted=True)
    c = FakeChannel([issue()], {4: [prompt(11, "the prompt"), down]})
    attend.once(c, conn=None)
    assert forks == ["9f3c1d20"]


def test_a_thumbs_down_on_my_prompt_means_the_same_point(
    turns: list[dict], found, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Undo this exchange, whichever half of it I happened to tap."""
    forks = rewound(monkeypatch)
    down = prompt(11, "forget I asked")
    down = channel.Comment(id=11, body=down.body, author="dandavison", rewind_wanted=True)
    c = FakeChannel([issue()], {4: [down, answered()]})
    attend.once(c, conn=None)
    assert forks == ["9f3c1d20"]


def test_the_rewound_exchange_is_deleted(
    turns: list[dict], found, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The transcript keeps it; the thread is only the live conversation."""
    rewound(monkeypatch)
    down = channel.Comment(id=12, body=answered().body, author="a[bot]", rewind_wanted=True)
    later = prompt(13, "and this came after")
    c = FakeChannel([issue()], {4: [prompt(11, "the prompt"), down, later]})
    attend.once(c, conn=None)
    assert set(c.deleted) == {11, 12, 13}


def test_the_issue_then_names_the_forked_session(
    turns: list[dict], found, monkeypatch: pytest.MonkeyPatch
) -> None:
    rewound(monkeypatch)
    down = channel.Comment(id=12, body=answered().body, author="a[bot]", rewind_wanted=True)
    c = FakeChannel([issue()], {4: [prompt(11, "the prompt"), down]})
    attend.once(c, conn=None)
    assert "claude:newnative" in c.bodies[4]


def test_a_rewind_does_not_also_run_the_prompt_it_undid(
    turns: list[dict], found, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It was just deleted. Answering it would be answering a question I withdrew."""
    rewound(monkeypatch)
    down = channel.Comment(id=11, body="forget I asked", author="dandavison", rewind_wanted=True)
    c = FakeChannel([issue()], {4: [down, answered()]})
    attend.once(c, conn=None)
    assert turns == []


def test_the_answer_records_the_point_it_can_be_rewound_to(turns: list[dict], found) -> None:
    """Without this the tap has nowhere to go, and the rewind is only a design."""
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert comment.point(c.edited[-1][1]) == "leafbefore"


def test_prompts_that_piled_up_are_all_still_waiting(turns: list[dict], found) -> None:
    """Three in a row while the loop was down, and the session has none of them.

    This used to be decided by counting replies in the thread, which needed the
    replies to arrive in step with the prompts and broke when they did not. The
    session knows what it has taken in.
    """
    piled = [prompt(11, "first"), prompt(12, "second"), prompt(13, "third")]
    assert [c.body for c in attend.pending(piled, [])] == ["first", "second", "third"]


def test_one_the_session_has_taken_in_is_no_longer_waiting(turns: list[dict], found) -> None:
    piled = [prompt(11, "first"), prompt(12, "second")]
    taken: list[Block] = [Said(role="user", text="first", uuid="u1")]
    assert [c.body for c in attend.pending(piled, taken)] == ["second"]


def test_a_stale_progress_comment_settles_nothing(turns: list[dict], found) -> None:
    """What a killed turn leaves. It is not a rendering, and it claims no prompt."""
    running = channel.Comment(id=99, body=f"{channel.RUNNING}\nworking…", author="a[bot]")
    assert [c.body for c in attend.pending([prompt(11, "first"), running], [])] == ["first"]


# --- what survives a crash ----------------------------------------------------


def test_the_hold_names_the_agent_once_it_has_been_spawned(monkeypatch, tmp_path) -> None:
    """The server's pid is not what has the transcript open; the agent's is."""
    named: list[tuple[str, int]] = []
    monkeypatch.setattr(attend.attending, "writer", lambda n, p: named.append((n, p)))
    monkeypatch.setattr(attend, "watch", lambda *a, **k: None)

    class Fake:
        pid = 4242
        returncode = 0

        def communicate(self, text, timeout=None):
            return "{}", ""

        def terminate(self) -> None:
            pass

    monkeypatch.setattr(attend.subprocess, "Popen", lambda *a, **k: Fake())
    monkeypatch.setattr(index.SOURCES["claude"], "leaf", lambda path: "")
    monkeypatch.setattr(index.SOURCES["claude"], "blocks", lambda path, since="": [])
    attend.run_turn(SESSION, "hello")
    assert named == [("7e90", 4242)]


def test_an_interrupted_turn_is_said_to_be_interrupted(
    turns: list[dict], found, monkeypatch
) -> None:
    """A half-answer passed off as an answer is the worst thing this can do."""
    monkeypatch.setattr(attend.attending, "interrupted", lambda native: True)
    c = FakeChannel([issue()], {4: [prompt()]})
    attend.once(c, conn=None)
    assert "cut off" in c.edited[-1][1].lower()


def test_a_pass_does_not_swallow_a_credential_failure(monkeypatch, capsys) -> None:
    """Everything else is worth retrying. This is worth stopping for."""

    class Denied(FakeChannel):
        def issues(self):
            raise channel.NotAuthorized("the token is not valid")

    with pytest.raises(channel.NotAuthorized):
        attend.a_pass(Denied([]), conn=None)


# --- the thread reconciled against the session --------------------------------


def transcript(monkeypatch, *blocks) -> None:
    monkeypatch.setattr(index.SOURCES["claude"], "blocks", lambda path, since="": list(blocks))


def said_by_me(text: str, uuid: str) -> Said:
    return Said(role="user", text=text, uuid=uuid)


def test_a_turn_i_had_at_the_keyboard_appears_in_the_thread(found, monkeypatch) -> None:
    """The guarantee the thread could not make: mixed use is now the normal case.

    Nothing was posted for work done in the TUI, so the thread read as though
    the conversation were only what came through it.
    """
    transcript(
        monkeypatch, said_by_me("asked at the keyboard", "u1"), Said("assistant", "answered")
    )
    c = FakeChannel([issue()], {4: []})
    attend.reconcile(c, issue(), SESSION)
    assert comment.turn_key(c.posted[0][1]) == "u1"
    assert "answered" in c.posted[0][1]


def test_a_turn_whose_rendering_has_changed_is_rewritten_not_duplicated(found, monkeypatch) -> None:
    transcript(monkeypatch, said_by_me("q", "u1"), Said("assistant", "the fuller answer"))
    already = channel.Comment(
        id=50, body=comment.render_turn(comment.Turn("u1", "q", [])), author="a[bot]"
    )
    c = FakeChannel([issue()], {4: [already]})
    attend.reconcile(c, issue(), SESSION)
    assert c.posted == []
    assert c.edited[0][0] == 50
    assert "the fuller answer" in c.edited[0][1]


def test_a_rendering_of_a_turn_that_is_gone_is_deleted(found, monkeypatch) -> None:
    """After a rewind the session is shorter, and the thread should be too."""
    transcript(monkeypatch)
    stale = channel.Comment(
        id=50, body=comment.render_turn(comment.Turn("u1", "q", [])), author="a[bot]"
    )
    c = FakeChannel([issue()], {4: [stale]})
    attend.reconcile(c, issue(), SESSION)
    assert c.deleted == [50]


def test_my_own_comments_are_never_touched(found, monkeypatch) -> None:
    """They are the input. Nothing here is entitled to rewrite what I asked."""
    transcript(monkeypatch)
    c = FakeChannel([issue()], {4: [prompt(11, "mine")]})
    attend.reconcile(c, issue(), SESSION)
    assert c.deleted == []
    assert c.edited == []


def test_a_progress_comment_is_cleared_once_nothing_is_running(found, monkeypatch) -> None:
    """Reconciling happens between turns, so any left over is litter from a crash."""
    transcript(monkeypatch)
    litter = channel.Comment(id=60, body=f"{channel.RUNNING}\nworking…", author="a[bot]")
    c = FakeChannel([issue()], {4: [litter]})
    attend.reconcile(c, issue(), SESSION)
    assert c.deleted == [60]


def test_a_prompt_already_in_the_session_is_not_run_again(found, monkeypatch) -> None:
    """Consumption records itself: the turn writes my prompt into the transcript.

    That is what makes at-most-once hold without bookkeeping. Counting replies
    in the thread was the bookkeeping, and it desynced every time the thread
    was edited.
    """
    ran: list[str] = []
    monkeypatch.setattr(
        attend, "run_turn", lambda s, t, showing=None, tag="": (ran.append(t), ([], {}, ""))[1]
    )
    transcript(monkeypatch, said_by_me("already asked", "u1"), Said("assistant", "done"))
    c = FakeChannel([issue()], {4: [prompt(11, "already asked")]})
    attend.once(c, conn=None)
    assert ran == []


def test_a_prompt_not_yet_in_the_session_is_run(turns: list[dict], found, monkeypatch) -> None:
    transcript(monkeypatch, said_by_me("something else", "u1"))
    c = FakeChannel([issue()], {4: [prompt(11, "brand new")]})
    attend.once(c, conn=None)
    assert turns[0]["prompt"] == "brand new"


def test_a_prompt_a_rendering_says_it_answered_is_not_run_again(
    turns: list[dict], found, monkeypatch
) -> None:
    """The second guard, for when the text does not match exactly.

    Running the same prompt twice is the only thing here that cannot be undone,
    so it is worth two independent reasons not to.
    """
    transcript(monkeypatch)
    claimed = channel.Comment(
        id=50,
        body=comment.render_turn(comment.Turn("u1", "q", []), asked_by=11),
        author="a[bot]",
    )
    c = FakeChannel([issue()], {4: [prompt(11, "asked"), claimed]})
    attend.once(c, conn=None)
    assert turns == []


def test_a_prompt_is_marked_done_when_its_answer_lands(turns: list[dict], found) -> None:
    """Eyes on means working; a rocket means the answer is there."""
    c = FakeChannel([issue()], {4: [prompt(11)]})
    attend.once(c, conn=None)
    assert c.taken == [11]
    assert c.done == [11]


def test_the_same_prompt_sent_twice_runs_once(turns: list[dict], found, monkeypatch) -> None:
    """A double send from a phone, and both of them ran.

    Pending was worked out once, before any turn, so the second copy was still
    on the list after the first had put that very text into the session. The
    rule was right; it was being asked at the wrong moment.
    """
    consumed: list[Block] = []

    def fake(session, text, showing=None, tag=""):
        consumed.append(Said(role="user", text=text, uuid=f"u{len(consumed)}"))
        return [Said(role="assistant", text="Done.")], {}, "before"

    monkeypatch.setattr(attend, "run_turn", fake)
    monkeypatch.setattr(index.SOURCES["claude"], "blocks", lambda path, since="": list(consumed))
    same = "Can we view this as a matrix"
    c = FakeChannel([issue()], {4: [prompt(11, same), prompt(12, same)]})
    attend.once(c, conn=None)
    assert [b.text for b in consumed if isinstance(b, Said)] == [same]


def test_two_different_prompts_in_one_pass_both_run(turns: list[dict], found, monkeypatch) -> None:
    """Re-checking must not turn a queue of real prompts into a queue of one."""
    consumed: list[Block] = []

    def fake(session, text, showing=None, tag=""):
        consumed.append(Said(role="user", text=text, uuid=f"u{len(consumed)}"))
        return [Said(role="assistant", text="Done.")], {}, "before"

    monkeypatch.setattr(attend, "run_turn", fake)
    monkeypatch.setattr(index.SOURCES["claude"], "blocks", lambda path, since="": list(consumed))
    c = FakeChannel([issue()], {4: [prompt(11, "first"), prompt(12, "second")]})
    attend.once(c, conn=None)
    assert [b.text for b in consumed if isinstance(b, Said)] == ["first", "second"]
