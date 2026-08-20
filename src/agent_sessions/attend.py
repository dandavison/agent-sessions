"""Attend the control channel: notice a prompt, run the turn, post what came back.

This is the only part of the tool that runs an agent with nobody watching, so
what it refuses is as much of the design as what it does. A prompt is taken up
once. A session already open in a terminal is left alone, because two agents on
one transcript would corrupt it. An issue with no frontmatter is not an
invitation to run anything.

A turn appends to the same transcript that an interactive one would, so
everything downstream — `sync`, `search`, `show`, the pages `serve` puts up —
sees this work exactly as it sees the rest. Nothing here is a separate history.
"""

import subprocess
import time
from typing import Any, Protocol

import orjson

from agent_sessions import channel, comment, index, query

# What a prompt from the park may do. A bare `Bash` is every command there is,
# so it is not on the list; the commands that are, are named. Editing is here
# because reading alone is not working, and a worktree with git in it is the
# undo. Anything else fails and says so, which is the whole of the policy until
# the approval loop lands.
ALLOWED = [
    "Read",
    "Grep",
    "Glob",
    "NotebookRead",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Edit",
    "Write",
    "MultiEdit",
    "Bash(git status:*)",
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Bash(git show:*)",
    "Bash(git branch:*)",
    "Bash(ls:*)",
    "Bash(rg:*)",
    "Bash(fd:*)",
    "Bash(uv run pytest:*)",
    "Bash(uv run ruff:*)",
    "Bash(uv run ty:*)",
]

INTERVAL = 5.0

# Long enough for a real turn, short enough that a wedged one is noticed.
TIMEOUT = 1800.0


class Control(Protocol):
    """What the loop needs of a channel, so a fake is a fake and not a mock."""

    def issues(self) -> list[channel.Issue]: ...
    def comments(self, number: int) -> list[channel.Comment]: ...
    def post(self, number: int, body: str) -> None: ...
    def take_up(self, comment_id: int) -> None: ...
    def set_body(self, number: int, body: str) -> None: ...


def loop(control: Control, conn: Any, interval: float = INTERVAL) -> None:
    while True:
        once(control, conn)
        time.sleep(interval)


def once(control: Control, conn: Any) -> int:
    """One pass over every open issue. Returns how many prompts were answered."""
    answered = 0
    for issue in control.issues():
        if not issue.session_id:
            continue
        answered += _attend(control, conn, issue)
    return answered


def _attend(control: Control, conn: Any, issue: channel.Issue) -> int:
    waiting = [c for c in control.comments(issue.number) if not c.is_ours and not c.taken_up]
    if not waiting:
        return 0
    session = lookup(conn, issue.session_id)
    answered = 0
    for prompt in waiting:
        # The eyes first: a turn takes minutes, and without them it looks from
        # the park like the prompt fell on the floor.
        control.take_up(prompt.id)
        if refusal := _refusal(session, issue.session_id):
            control.post(issue.number, refusal)
            continue
        assert session is not None
        control.post(issue.number, comment.render(run_turn(session, prompt.body, ALLOWED)))
        answered += 1
    _restate(control, issue, session)
    return answered


def _refusal(session: dict[str, Any] | None, session_id: str) -> str:
    if session is None:
        return f"No session matches `{session_id}`. Fix the table in the issue body."
    running = index.SOURCES[session["agent"]].live().get(session["native_id"])
    if running:
        return (
            f"`{session_id}` is running in a terminal ({running.status}), so I have left it"
            " alone: two agents on one transcript would corrupt it."
        )
    return ""


def _restate(control: Control, issue: channel.Issue, session: dict[str, Any] | None) -> None:
    """Keep the body current: it is the only place my turns appear without the agent's."""
    prompts = [c.body for c in control.comments(issue.number) if not c.is_ours]
    control.set_body(issue.number, comment.body(session or {"id": issue.session_id}, prompts))


def lookup(conn: Any, session_id: str) -> dict[str, Any] | None:
    return query.get(conn, session_id)


def run_turn(session: dict[str, Any], prompt: str, allowed: list[str]) -> list[dict[str, Any]]:
    """One headless turn, in the directory the session was had in.

    It appends to the session's own transcript, so this is the same session
    continuing and not a copy of it.
    """
    source = index.SOURCES[session["agent"]]
    done = subprocess.run(
        source.turn_command(session["native_id"], allowed),
        cwd=session["cwd"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )
    return _events(done.stdout) or [_failed(done)]


def _events(stdout: str) -> list[dict[str, Any]]:
    return [orjson.loads(line) for line in stdout.splitlines() if line.startswith("{")]


def _failed(done: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Nothing came back. Say so in the thread rather than nowhere."""
    detail = (done.stderr or "no output").strip()[:2_000]
    text = f"The turn produced nothing (exit {done.returncode}).\n\n```\n{detail}\n```"
    return {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}}
