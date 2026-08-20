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
