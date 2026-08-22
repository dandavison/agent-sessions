"""A hold on a session, for the turns an agent does not register itself.

Claude writes a file under `~/.claude/sessions` for an interactive session and
nothing at all for a headless one, so `live()` — the check that stops a second
agent being started on a transcript that already has one — cannot see a turn
run from the control channel. Resuming in a terminal while the park is mid-turn
would put two writers on one file.

So the hold is ours to keep. It is local, unlike the marker and the eyes, which
are about a conversation and belong on GitHub: this is about which process on
this machine has the transcript open.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DIR = Path.home() / ".agent-sessions" / "attending"

# One loop at a time. Two of them both see the same unconsumed prompt and both
# run it, and each one's takeover kills the other's turn, since a headless turn
# registers as live. At-most-once cannot hold while a second exists.
LOCK = Path.home() / ".agent-sessions" / "attending.pid"


class AlreadyAttending(Exception):
    def __init__(self, pid: str) -> None:
        super().__init__(f"Another attend loop is running (pid {pid}).")


@contextmanager
def only_one() -> Iterator[None]:
    """Hold the right to be the attend loop, and give it back on the way out.

    A lock left by a loop that died is not a lock, so a crash never needs
    clearing up by hand before the thing will run again.
    """
    if LOCK.exists() and _alive(held_by := LOCK.read_text().strip()):
        raise AlreadyAttending(held_by)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(str(os.getpid()))
    try:
        yield
    finally:
        LOCK.unlink(missing_ok=True)


@contextmanager
def holding(native_id: str, pid: int | None = None) -> Iterator[None]:
    """Hold a session for the length of a turn, whatever the turn does.

    The pid recorded is the one with the transcript open — the agent, not
    whatever spawned it. Naming the parent was wrong in the way that matters:
    kill the server and its agent is reparented and keeps writing, while the
    recorded pid is dead, so the hold lifts and a fresh server can start a
    second turn on a file that already has a writer.
    """
    path = _path(native_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid if pid is not None else os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def writer(native_id: str, pid: int) -> None:
    """Name the process that actually has the file, once it has been spawned."""
    path = _path(native_id)
    if path.exists():
        path.write_text(str(pid))


def interrupted(native_id: str) -> bool:
    """Whether a turn was cut off, and clear the mark by asking.

    A hold is removed when a turn ends, so one whose process has gone says a
    turn started and never finished. That is the single fact the transcript
    cannot express on its own — it holds the work, not the intent to finish —
    and it is what the thread has to say rather than passing a half-answer off
    as an answer.
    """
    path = _path(native_id)
    if not path.exists():
        return False
    if _alive(path.read_text().strip()):
        return False
    path.unlink(missing_ok=True)
    return True


def held(native_id: str) -> bool:
    """Whether a turn is running on this session right now.

    A hold left behind by a process that has gone is not a hold: a loop that
    was killed must not leave a session unresumable until the next reboot.
    """
    path = _path(native_id)
    if not path.exists():
        return False
    if _alive(path.read_text().strip()):
        return True
    path.unlink(missing_ok=True)
    return False


def _path(native_id: str) -> Path:
    return DIR / native_id


def _alive(pid: str) -> bool:
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, OverflowError, ValueError):
        return False
    except PermissionError:
        return True
    return True
