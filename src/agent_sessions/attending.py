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


@contextmanager
def holding(native_id: str) -> Iterator[None]:
    """Hold a session for the length of a turn, whatever the turn does."""
    path = _path(native_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


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
