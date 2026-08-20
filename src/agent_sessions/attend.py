"""Attend the control channel: notice a prompt, run the turn, post what came back.

This is the only part of the tool that runs an agent with nobody watching, so
what it refuses is as much of the design as what it does. A prompt is taken up
once. A session already open in a terminal is left alone, because two agents on
one transcript would corrupt it. An issue with no frontmatter is not an
invitation to run anything.

A turn appends to the same transcript that an interactive one would, so
everything downstream — `sync`, `search`, `show`, the pages `serve` puts up —
sees this work exactly as it sees the rest. Nothing here is a separate history.

A turn runs under my own settings, so a prompt left here is allowed whatever a
prompt typed at the keyboard is allowed. Nothing stands between a comment on
the issue and this machine except who can reach the repo.

A session open in a pane at home is taken over rather than refused. I am not at
that keyboard, and a refusal is not a tool working.
"""

import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

import orjson

from agent_sessions import attending, channel, comment, index, query
from agent_sessions.models import Block, Said

INTERVAL = 5.0

# Long enough for an agent to put its affairs in order on SIGTERM.
TAKEOVER_WAIT = 10.0

# Long enough for a real turn, short enough that a wedged one is noticed.
TIMEOUT = 1800.0


class Control(Protocol):
    """What the loop needs of a channel, so a fake is a fake and not a mock."""

    def issues(self) -> list[channel.Issue]: ...
    def comments(self, number: int) -> list[channel.Comment]: ...
    def post(self, number: int, body: str) -> None: ...
    def take_up(self, comment_id: int) -> None: ...
    def set_body(self, number: int, body: str) -> None: ...
    def open(self, title: str, body: str) -> channel.Issue: ...


def issue_for(control: Control, session: dict[str, Any]) -> channel.Issue:
    """The issue that makes this session reachable from a phone, opening one if needed.

    Which session an issue is for is written in its body and read back from
    there, so this asks GitHub rather than keeping a mapping of its own. The
    index is safe to delete; the answer to `is there already an issue for this`
    must not be.
    """
    for existing in control.issues():
        if existing.session_id == session["id"]:
            return existing
    return control.open(session.get("title") or session["id"], comment.body(session, []))


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
        if session is None:
            control.post(
                issue.number,
                f"No session matches `{issue.session_id}`. Fix the table in the issue body.",
            )
            continue
        displaced = _clear_the_way(session)
        # Held for the length of the turn, so a resume from the terminal is
        # refused rather than opening a second agent on the same transcript.
        with attending.holding(session["native_id"]):
            blocks, summary = run_turn(session, prompt.body)
        control.post(issue.number, displaced + comment.render(blocks, summary))
        answered += 1
    _restate(control, issue, session)
    return answered


def _clear_the_way(session: dict[str, Any]) -> str:
    """Take the session off whatever else holds it, and say so.

    Two agents on one transcript fork it and then fight over the `last-prompt`
    record that says which branch is live. One of them has to go, and it is not
    the one I am talking to: I am not at that keyboard, and the pane holds
    nothing the transcript does not.
    """
    running = index.SOURCES[session["agent"]].live().get(session["native_id"])
    if not running:
        return ""
    take_over(running.pid)
    return (
        f"<sub>Taken over from a terminal ({running.status});"
        " resume it to pick it back up.</sub>\n\n"
    )


def take_over(pid: int) -> None:
    """Stop the process holding a session, and wait until it has actually gone."""
    with suppress(ProcessLookupError):
        os.kill(pid, signal.SIGTERM)
    for _ in range(int(TAKEOVER_WAIT / 0.1)):
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            return
        time.sleep(0.1)


def _restate(control: Control, issue: channel.Issue, session: dict[str, Any] | None) -> None:
    """Keep the body current: it is the only place my turns appear without the agent's."""
    prompts = [c.body for c in control.comments(issue.number) if not c.is_ours]
    control.set_body(issue.number, comment.body(session or {"id": issue.session_id}, prompts))


def lookup(conn: Any, session_id: str) -> dict[str, Any] | None:
    return query.get(conn, session_id)


def run_turn(session: dict[str, Any], prompt: str) -> tuple[list[Block], dict[str, Any]]:
    """One headless turn, in the directory the session was had in.

    A turn appends to the session's own transcript, so what it said is read
    back from there rather than parsed off stdout: one parser, and the comment
    cannot say something different from what `cat` and the pages say. Only what
    the turn cost comes off stdout, because the transcript does not hold it.
    """
    source = index.SOURCES[session["agent"]]
    path = Path(session["path"])
    before = source.leaf(path)
    done = subprocess.run(
        source.turn_command(session["native_id"]),
        cwd=session["cwd"],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
        check=False,
    )
    blocks = source.blocks(path, since=before)
    return (blocks or [_failed(done)]), _summary(done.stdout)


def _summary(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    return orjson.loads(text) if text.startswith("{") else {}


def _failed(done: subprocess.CompletedProcess[str]) -> Said:
    """The transcript grew by nothing. Say so in the thread rather than nowhere."""
    detail = (done.stderr or "no output").strip()[:2_000]
    return Said(
        role="assistant",
        text=f"The turn added nothing (exit {done.returncode}).\n\n```\n{detail}\n```",
    )
