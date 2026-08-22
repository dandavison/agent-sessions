"""Who currently holds a session, when the holder is not a terminal.

An agent registers a running interactive session under `~/.claude/sessions`, so
`live()` can see one and refuse to start a second on the same transcript. It
registers nothing for a headless turn — measured, not assumed — so a turn run
from the control channel is invisible to exactly the check that exists to stop
two agents corrupting one transcript.

Hence a lock of our own. It is local because it is about processes on this
machine, unlike the marker and the eyes, which are about a conversation and
live on GitHub.
"""

import os
from pathlib import Path

import pytest

from agent_sessions import attending, resume, wormhole

NATIVE = "7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"


@pytest.fixture
def locks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(attending, "DIR", tmp_path / "attending")
    monkeypatch.setattr(attending, "LOCK", tmp_path / "attending.pid")
    return tmp_path / "attending"


def test_nobody_holds_a_session_nobody_is_running(locks: Path) -> None:
    assert not attending.held(NATIVE)


def test_a_turn_in_progress_is_held(locks: Path) -> None:
    with attending.holding(NATIVE):
        assert attending.held(NATIVE)


def test_the_hold_is_released_when_the_turn_ends(locks: Path) -> None:
    with attending.holding(NATIVE):
        pass
    assert not attending.held(NATIVE)


def test_the_hold_is_released_even_when_the_turn_raises(locks: Path) -> None:
    """A turn that dies must not lock the session out until the next reboot."""
    with pytest.raises(RuntimeError), attending.holding(NATIVE):
        raise RuntimeError("boom")
    assert not attending.held(NATIVE)


def test_a_hold_left_by_a_process_that_died_is_not_a_hold(locks: Path) -> None:
    """A killed loop must not leave a session unresumable forever."""
    locks.mkdir(parents=True)
    (locks / NATIVE).write_text("2147483647")  # not a pid that exists
    assert not attending.held(NATIVE)


def test_a_hold_by_a_process_still_alive_is_a_hold(locks: Path) -> None:
    locks.mkdir(parents=True)
    (locks / NATIVE).write_text(str(os.getpid()))
    assert attending.held(NATIVE)


# --- what the hold is for -----------------------------------------------------


def test_a_session_mid_turn_is_not_resumed(
    locks: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hazard the lock exists for: `live()` cannot see the turn that is running.

    Asserted on the handoff rather than only on the message: a path can contain
    any word, and this test once passed because pytest's own temporary
    directory is named after it.
    """
    called: list[dict] = []
    monkeypatch.setattr(wormhole, "run", lambda **kw: called.append(kw))
    cwd = tmp_path / "src" / "wormhole"
    cwd.mkdir(parents=True)
    session = {
        "id": f"claude:{NATIVE}",
        "agent": "claude",
        "native_id": NATIVE,
        "path": str(cwd / f"{NATIVE}.jsonl"),
        "cwd": str(cwd),
        "project": "wormhole",
    }
    Path(session["path"]).write_text("{}\n")
    with attending.holding(NATIVE), pytest.raises(resume.NotResumable, match="already running"):
        resume.resume(session)
    assert called == []


# --- whose pid the hold records, and why it matters --------------------------


def test_the_hold_names_the_process_doing_the_writing(locks: Path) -> None:
    """The server's pid is the wrong one: it is not what has the transcript open.

    SIGKILL the server and the agent it spawned is reparented and keeps
    writing. The hold's pid is then dead, so it stops holding, and a fresh
    server starts a second turn on a transcript that already has a writer.
    """
    with attending.holding(NATIVE, pid=4242):
        assert (locks / NATIVE).read_text().strip() == "4242"


def test_a_hold_outlives_the_process_that_took_it(
    locks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a crashed server leaves behind: the turn is still going."""
    locks.mkdir(parents=True, exist_ok=True)
    (locks / NATIVE).write_text(str(os.getpid()))
    assert attending.held(NATIVE)


def test_a_hold_left_by_a_finished_turn_is_the_mark_of_an_interruption(locks: Path) -> None:
    """A stale hold is the one fact the transcript cannot express on its own.

    Removed when a turn ends, so finding one whose process has gone means the
    turn was cut off — which is what the thread has to say rather than passing
    a half-answer off as an answer.
    """
    locks.mkdir(parents=True, exist_ok=True)
    (locks / NATIVE).write_text("2147483647")
    assert attending.interrupted(NATIVE)
    assert not attending.held(NATIVE)
    assert not attending.interrupted(NATIVE)  # reading it clears it


# --- only one loop, and no agent left writing after it goes ------------------


def test_a_second_loop_refuses_rather_than_fighting_the_first(locks: Path) -> None:
    """Two loops both see the same unconsumed prompt and both run it.

    Worse, each one's takeover kills the other's turn, because a headless turn
    registers as live. At-most-once cannot hold while two of these exist.
    """
    with attending.only_one(), pytest.raises(attending.AlreadyAttending), attending.only_one():
        pass


def test_the_loop_may_start_again_once_the_first_has_gone(locks: Path) -> None:
    with attending.only_one():
        pass
    with attending.only_one():
        assert True


def test_a_lock_from_a_dead_loop_does_not_block_a_new_one(locks: Path) -> None:
    """A crash must not need a manual cleanup before the thing runs again."""
    attending.LOCK.parent.mkdir(parents=True, exist_ok=True)
    attending.LOCK.write_text("2147483647")
    with attending.only_one():
        assert True
