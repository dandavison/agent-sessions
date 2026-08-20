"""What a long-running command says about itself while nobody is watching.

`attend` runs for hours and then I come back to a terminal and want four things
from it: whether it is alive, whether it saw what I sent, what it is doing now,
and what went wrong. Every line is stamped, because "it hung at some point" is
not something I can act on.

stderr, like every other word this tool says that is not data.
"""

import sys
from datetime import datetime

# A poll every five seconds would bury everything worth reading, so the
# per-poll detail is asked for rather than endured.
VERBOSE: bool = False

PROBLEM = "!!"


def say(message: str) -> None:
    """Something happened that I would want to see in a scrollback."""
    _write(message)


def detail(message: str) -> None:
    """Something happened that is only interesting when something is wrong."""
    if VERBOSE:
        _write(message)


def problem(message: str) -> None:
    """Something went wrong. Marked so it is findable by eye in a wall of lines."""
    _write(f"{PROBLEM} {message}")


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _write(message: str) -> None:
    print(f"{stamp()} {message}", file=sys.stderr, flush=True)
