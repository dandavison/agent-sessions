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
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from threading import Thread
from typing import Any, Protocol

import orjson

from agent_sessions import attending, channel, comment, index, log, query
from agent_sessions.models import Block, Ran, Said

INTERVAL = 5.0

# How often the transcript is looked at while a turn runs, and how long it may
# say nothing before saying that it is alive.
POLL = 1.0
HEARTBEAT = 20.0

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
        a_pass(control, conn)
        time.sleep(interval)


def a_pass(control: Control, conn: Any) -> int:
    """One pass, and whatever went wrong in it is not the end of the loop.

    This runs for hours while I am out. A blip at GitHub, a turn that dies, a
    transcript that moved — each is a reason to say so and poll again, not to
    stop answering until I get home.
    """
    try:
        return once(control, conn)
    except Exception as e:  # noqa: BLE001 — the loop outliving the pass is the point
        log.problem(f"{type(e).__name__}: {e}")
        return 0


def once(control: Control, conn: Any) -> int:
    """One pass over every open issue. Returns how many prompts were answered."""
    answered = 0
    issues = control.issues()
    log.detail(f"polled: {len(issues)} open")
    for issue in issues:
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
        log.say(f"#{issue.number} picked up: {_oneline(prompt.body)}")
        # The eyes first: a turn takes minutes, and without them it looks from
        # the park like the prompt fell on the floor.
        control.take_up(prompt.id)
        if session is None:
            log.problem(f"#{issue.number} names {issue.session_id}, which is not a session")
            control.post(
                issue.number,
                f"No session matches `{issue.session_id}`. Fix the table in the issue body.",
            )
            continue
        displaced = _clear_the_way(session)
        started = time.monotonic()
        # Held for the length of the turn, so a resume from the terminal is
        # refused rather than opening a second agent on the same transcript.
        with attending.holding(session["native_id"]):
            blocks, summary = run_turn(session, prompt.body)
        log.say(
            f"#{issue.number} answered in {time.monotonic() - started:.0f}s"
            f" — {len(blocks)} blocks, ${summary.get('total_cost_usd', 0):.2f}"
        )
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
    log.say(f"taking {session['id']} over from pid {running.pid} ({running.status})")
    take_over(running.pid)
    return (
        f"<sub>Taken over from a terminal ({running.status});"
        " resume it to pick it back up.</sub>\n\n"
    )


def _oneline(text: str) -> str:
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first if len(first) <= 80 else first[:77] + "…"


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
    done = _spawn(
        source.turn_command(session["native_id"]),
        cwd=session["cwd"],
        prompt=prompt,
        watching=lambda: source.blocks(path, since=before),
    )
    blocks = source.blocks(path, since=before)
    return (blocks or [_failed(done)]), _summary(done.stdout)


def _spawn(
    argv: list[str], cwd: str, prompt: str, watching: Callable[[], list[Block]]
) -> subprocess.CompletedProcess[str]:
    """Run the turn, saying what it does while it does it.

    The process says nothing until it exits, so the progress comes off the
    transcript it is appending to. Talking to the process is on its own thread
    only so that this one is free to watch the file.
    """
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    spoken: dict[str, str] = {}

    def talk() -> None:
        spoken["out"], spoken["err"] = proc.communicate(prompt, timeout=TIMEOUT)

    worker = Thread(target=talk, daemon=True)
    worker.start()
    watch(watching, alive=worker.is_alive)
    worker.join(TIMEOUT)
    if worker.is_alive():
        proc.kill()
        worker.join(5)
        log.problem(f"the turn passed {TIMEOUT:.0f}s and was killed")
    return subprocess.CompletedProcess(
        argv, proc.returncode or 0, spoken.get("out", ""), spoken.get("err", "")
    )


def watch(blocks: Callable[[], list[Block]], alive: Callable[[], bool]) -> None:
    """Report what the turn has done so far, until it stops running.

    Said as it lands rather than at the end, because the question while waiting
    is always whether the thing is still moving.
    """
    reported = 0
    quiet = time.monotonic()
    while alive():
        for block in blocks()[reported:]:
            log.say(f"  {_progress(block)}")
            reported += 1
            quiet = time.monotonic()
        if time.monotonic() - quiet >= HEARTBEAT:
            log.say(f"  still going ({reported} so far)")
            quiet = time.monotonic()
        time.sleep(POLL)


def _progress(block: Block) -> str:
    match block:
        case Ran():
            return f"{block.tool} {_oneline(str(next(iter(block.input.values()), '')))}"
        case Said():
            return _oneline(block.text)
        case _:
            return "compacted"


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
