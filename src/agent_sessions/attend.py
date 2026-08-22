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

import atexit
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

# A poll that finds nothing returns 304 and costs no rate limit, so this is
# bounded by politeness rather than budget: one pass is a request for the issue
# list plus one per open issue.
# Every agent this loop has spawned, so that stopping the loop stops them too.
# An orphan keeps writing to a transcript nobody thinks is being written to.
_children: set[subprocess.Popen[str]] = set()

# How many turns an issue shows when it does not say where to start. The old
# ones say nothing, and a whole session is neither readable nor cheap.
MOST = 20

INTERVAL = 2.0

# How often the transcript is looked at while a turn runs, and how long it may
# say nothing before saying that it is alive.
POLL = 1.0
HEARTBEAT = 20.0

# How often the comment on the page is rewritten. An edit is a request, and a
# request per block would be one a second.
SHOW = 10.0

# Long enough for an agent to put its affairs in order on SIGTERM.
TAKEOVER_WAIT = 10.0

# Long enough for a real turn, short enough that a wedged one is noticed.
TIMEOUT = 1800.0


class Control(Protocol):
    """What the loop needs of a channel, so a fake is a fake and not a mock."""

    def issues(self) -> list[channel.Issue]: ...
    def comments(self, number: int) -> list[channel.Comment]: ...
    def post(self, number: int, body: str) -> int: ...
    def edit(self, comment_id: int, body: str) -> None: ...
    def delete(self, comment_id: int) -> None: ...
    def take_up(self, comment_id: int) -> None: ...
    def mark_done(self, comment_id: int) -> None: ...
    def set_body(self, number: int, body: str) -> None: ...
    def open(self, title: str, body: str) -> channel.Issue: ...


def issue_for(control: Control, session: dict[str, Any]) -> channel.Issue:
    """The issue that makes this session reachable from a phone, opening one if needed.

    A new one records where the session has got to, so the thread carries the
    conversation from then on rather than re-staging everything said before it.

    Which session an issue is for is written in its body and read back from
    there, so this asks GitHub rather than keeping a mapping of its own. The
    index is safe to delete; the answer to `is there already an issue for this`
    must not be.
    """
    for existing in control.issues():
        if existing.session_id == session["id"]:
            return existing
    started = index.SOURCES[session["agent"]].leaf(Path(session["path"]))
    return control.open(
        session.get("title") or session["id"], comment.body(session | {"from": started}, [])
    )


def loop(control: Control, conn: Any, interval: float = INTERVAL) -> None:
    """Attend the channel until interrupted, and take nothing with us on the way out."""
    with attending.only_one():
        atexit.register(stop_children)
        try:
            while True:
                a_pass(control, conn)
                time.sleep(interval)
        finally:
            stop_children()


def stop_children() -> None:
    """Stop any agent this loop spawned. An orphan writes where nobody is looking."""
    for child in list(_children):
        with suppress(ProcessLookupError):
            child.terminate()
        _children.discard(child)


def a_pass(control: Control, conn: Any) -> int:
    """One pass, and whatever went wrong in it is not the end of the loop.

    This runs for hours while I am out. A blip at GitHub, a turn that dies, a
    transcript that moved — each is a reason to say so and poll again, not to
    stop answering until I get home.
    """
    try:
        return once(control, conn)
    except channel.NotAuthorized:
        raise
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
    if _rewind(control, conn, issue):
        return 0
    session = lookup(conn, issue.session_id)
    waiting = pending(control.comments(issue.number), _seen(session))
    if not waiting:
        reconcile(control, issue, session)
        return 0
    log.say(f"#{issue.number} {len(waiting)} waiting")
    answered = 0
    for prompt in waiting:
        if prompt not in pending(control.comments(issue.number), _seen(session)):
            log.say(f"#{issue.number} skipping {prompt.id}: the session has it already")
            control.mark_done(prompt.id)
            continue
        tag = f"#{issue.number}·{prompt.id}"
        log.say(f"{tag} picked up: {_oneline(prompt.body)}")
        # The eyes first: a turn takes minutes, and without them it looks from
        # the park like the prompt fell on the floor.
        control.take_up(prompt.id)
        log.detail(f"{tag} eyes on")
        if session is None:
            log.problem(f"{tag} names {issue.session_id}, which is not a session")
            control.post(
                issue.number,
                f"No session matches `{issue.session_id}`. Fix the table in the issue body.",
            )
            continue
        displaced = _clear_the_way(session)
        cut_off = attending.interrupted(session["native_id"])
        log.say(f"{tag} running {session['id']} in {session['cwd']}")
        started = time.monotonic()
        # Before the turn, not after: the thread should show something within
        # seconds of asking, and this is the comment the answer will replace.
        note = control.post(issue.number, comment.working(prompt.body))
        log.say(f"{tag} working, in comment {note}")
        # Held for the length of the turn, so a resume from the terminal is
        # refused rather than opening a second agent on the same transcript.
        with attending.holding(session["native_id"]):
            blocks, summary, before = run_turn(
                session, prompt.body, showing=_shown(control, note, started, tag), tag=tag
            )
        log.say(
            f"{tag} answered in {time.monotonic() - started:.0f}s"
            f" — {len(blocks)} blocks, ${summary.get('total_cost_usd', 0):.2f}"
        )
        body = (
            channel.MARKER
            + "\n"
            + displaced
            + _cut_off_note(cut_off)
            + comment.render(blocks, summary, at_uuid=before)
        )
        control.edit(note, body)
        log.say(f"{tag} posted {len(body):,} chars")
        control.mark_done(prompt.id)
        answered += 1
    reconcile(control, issue, session)
    _restate(control, issue, session)
    return answered


def _shown(control: Control, note: int, started: float, tag: str) -> Callable[[list[Block]], None]:
    """Write what has happened so far into the comment already on the page.

    An edit does not notify, so this costs nothing but a request; the one
    notification is the comment appearing when the turn began.
    """

    def show(blocks: list[Block]) -> None:
        control.edit(note, comment.working("", blocks, time.monotonic() - started))
        log.detail(f"{tag} showed {len(blocks)} blocks")

    return show


def _rewind(control: Control, conn: Any, issue: channel.Issue) -> bool:
    """Put the session back the way it was before whichever exchange I thumbed down.

    A tap on my prompt and a tap on the answer to it mean the same thing, so
    both resolve to the point the answer recorded. Nothing is destroyed: the
    fork writes a new session and leaves the old one whole, which is why the
    thread can drop what was rewound over — the transcript still has it.
    """
    comments = control.comments(issue.number)
    asked = next((i for i, c in enumerate(comments) if c.rewind_wanted), None)
    if asked is None:
        return False
    log.say(f"#{issue.number} rewind asked for at comment {comments[asked].id}")
    # An exchange is my prompt and the answer to it. Tapping the answer means
    # undoing both, so the deleting starts at the prompt either way.
    start = asked
    while start > 0 and comments[start].is_ours:
        start -= 1
    at_uuid = next((c.point for c in comments[start:] if c.point), "")
    session = lookup(conn, issue.session_id)
    if session is None or not at_uuid:
        log.problem(f"#{issue.number} cannot rewind: no point recorded for that exchange")
        control.post(issue.number, f"{comment.MARKER}\nNothing here records where to rewind to.")
        return True

    source = index.SOURCES[session["agent"]]
    native = source.fork_at(Path(session["path"]), at_uuid)
    forked = f"{session['agent']}:{native}"
    log.say(f"#{issue.number} rewound to {at_uuid[:8]}, now {forked}")
    reindex(conn)
    log.detail(f"#{issue.number} reindexed after the fork")
    for gone in comments[start:]:
        control.delete(gone.id)
    log.say(f"#{issue.number} deleted {len(comments[start:])} comments rewound over")
    control.set_body(issue.number, comment.body(session | {"id": forked}, _said(session)))
    return True


def reindex(conn: Any) -> None:
    """A fork is a session nothing has read yet, and lookup reads the index."""
    index.sync(conn)


def reconcile(control: Control, issue: channel.Issue, session: dict[str, Any] | None) -> None:
    """Make the thread show the session, rather than whatever got posted.

    The thread was accumulated: the loop appended as it went, and every state
    it needed afterwards had to be inferred back out of the shape of it. Every
    bug worth the name came from that inference. Rendered instead, from the
    transcript, the thread cannot disagree with the session — and work done at
    the keyboard shows up without anything having to post it.

    My own comments are the input and are never touched. What is left over from
    a crash is: reconciling happens between turns, so a comment saying a turn is
    running is litter by definition.
    """
    if session is None:
        return
    blocks = index.SOURCES[session["agent"]].blocks(Path(session["path"]), since=window(issue))
    want = {t.key: comment.render_turn(t) for t in comment.turns(blocks)[-MOST:] if t.key}
    comments = control.comments(issue.number)
    have = {comment.turn_key(c.body): c for c in comments if c.is_ours and comment.turn_key(c.body)}
    made = changed = gone = 0
    for key, body in want.items():
        if key not in have:
            control.post(issue.number, body)
            made += 1
        elif have[key].body != body:
            control.edit(have[key].id, body)
            changed += 1
    for key, stale in have.items():
        if key not in want:
            control.delete(stale.id)
            gone += 1
    for litter in comments:
        if litter.is_progress:
            control.delete(litter.id)
            gone += 1
    if made or changed or gone:
        log.say(f"#{issue.number} reconciled: {made} posted, {changed} rewritten, {gone} deleted")


def window(issue: channel.Issue) -> str:
    """Where this issue's view of the session starts, if it says.

    An issue need not carry a session from its beginning — a long one would
    make an unreadable thread and a slow reconcile — so the body may name a
    point to start from.
    """
    return comment.frontmatter(issue.body).get("from", "")


def pending(comments: list[channel.Comment], blocks: list[Block]) -> list[channel.Comment]:
    """My comments that the session has not taken in yet.

    Asked of the transcript, not of the thread. A turn writes my prompt into
    the session as it starts, so the session already knows what it has
    consumed; inferring that from the shape of the thread is what desynced.

    A rendering that names the comment it answered settles it too, for the case
    where the text does not match exactly. Two reasons, because running a
    prompt twice is the only thing here that cannot be taken back.
    """
    taken_in = {_plain(b.text) for b in blocks if isinstance(b, Said) and b.role == "user"}
    claimed = {comment.asked_by(c.body) for c in comments if c.is_ours}
    return [
        c
        for c in comments
        if not c.is_ours and _plain(c.body) not in taken_in and c.id not in claimed
    ]


def _plain(text: str) -> str:
    """Whitespace is not meaning, and a comment box and a transcript disagree about it."""
    return " ".join(text.split())


def _seen(session: dict[str, Any] | None) -> list[Block]:
    """Everything the session holds, read fresh.

    All of it, not the window: the window says what the thread shows, and
    whether a prompt has been consumed is a fact about the session. Asked of
    the window, moving one forward re-armed every prompt behind it.

    Read again before each turn rather than once for the pass, because a pass
    consumes as it goes: a prompt sent twice was still on the list after the
    first copy had put that very text into the session.
    """
    if session is None:
        return []
    return index.SOURCES[session["agent"]].blocks(Path(session["path"]))


def _cut_off_note(cut_off: bool) -> str:
    """A half-answer passed off as an answer is the worst thing this can do."""
    if not cut_off:
        return ""
    return (
        "<sub>The turn before this one was cut off part way; what it had done is in"
        " the session. Say so if you want it carried on.</sub>\n\n"
    )


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
    log.say(f"pid {running.pid} has gone")
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
    """Keep the body current: the only place my turns appear without the agent's.

    Read off the transcript rather than off the thread. A prompt that never ran
    — interrupted, or posted while the loop was down — is on GitHub and not in
    the session, and the body should not assert a turn the agent has never
    heard of.
    """
    control.set_body(
        issue.number, comment.body(session or {"id": issue.session_id}, _said(session))
    )


def _said(session: dict[str, Any] | None) -> list[str]:
    if session is None:
        return []
    blocks = index.SOURCES[session["agent"]].blocks(Path(session["path"]))
    return [b.text for b in blocks if isinstance(b, Said) and b.role == "user"]


def lookup(conn: Any, session_id: str) -> dict[str, Any] | None:
    return query.get(conn, session_id)


def run_turn(
    session: dict[str, Any],
    prompt: str,
    showing: Callable[[list[Block]], None] | None = None,
    tag: str = "",
) -> tuple[list[Block], dict[str, Any], str]:
    """One headless turn, in the directory the session was had in.

    Returns where the transcript ended before it ran, too: that is the point a
    rewind of this turn goes back to, and only this knows it.

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
        native_id=session["native_id"],
        prompt=prompt,
        watching=lambda: source.blocks(path, since=before, tip=True),
        showing=showing,
        tag=tag,
    )
    blocks = source.blocks(path, since=before)
    return (blocks or [_failed(done)]), _summary(done.stdout), before


def _spawn(
    argv: list[str],
    cwd: str,
    native_id: str,
    prompt: str,
    watching: Callable[[], list[Block]],
    showing: Callable[[list[Block]], None] | None = None,
    tag: str = "",
) -> subprocess.CompletedProcess[str]:
    """Run the turn, saying what it does while it does it.

    The process says nothing until it exits, so the progress comes off the
    transcript it is appending to. Talking to the process is on its own thread
    only so that this one is free to watch the file.
    """
    log.detail(f"$ {' '.join(argv)}")
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    attending.writer(native_id, proc.pid)
    _children.add(proc)
    spoken: dict[str, str] = {}

    def talk() -> None:
        spoken["out"], spoken["err"] = proc.communicate(prompt, timeout=TIMEOUT)

    worker = Thread(target=talk, daemon=True)
    worker.start()
    watch(watching, alive=worker.is_alive, showing=showing, tag=tag)
    worker.join(TIMEOUT)
    _children.discard(proc)
    if worker.is_alive():
        proc.kill()
        worker.join(5)
        log.problem(f"the turn passed {TIMEOUT:.0f}s and was killed")
    return subprocess.CompletedProcess(
        argv, proc.returncode or 0, spoken.get("out", ""), spoken.get("err", "")
    )


def watch(
    blocks: Callable[[], list[Block]],
    alive: Callable[[], bool],
    showing: Callable[[list[Block]], None] | None = None,
    tag: str = "",
) -> None:
    """Report what the turn has done so far, until it stops running.

    Said as it lands rather than at the end, because the question while waiting
    is always whether the thing is still moving. `showing` puts the same thing
    where I am actually looking, which is not this terminal; it is throttled
    because a request per block would be a request per second.
    """
    reported = 0
    quiet = time.monotonic()
    told = 0.0
    while alive():
        seen = blocks()
        for block in seen[reported:]:
            log.say(f"{tag}   {_progress(block)}")
            reported += 1
            quiet = time.monotonic()
        if time.monotonic() - quiet >= HEARTBEAT:
            log.say(f"{tag}   still going ({reported} so far)")
            quiet = time.monotonic()
        if showing and time.monotonic() - told >= SHOW:
            showing(seen)
            told = time.monotonic()
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
