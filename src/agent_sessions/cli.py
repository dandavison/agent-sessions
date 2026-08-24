"""The agent-sessions command line."""

import sqlite3
import sys
import time
import webbrowser
from collections.abc import Callable, Iterator
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from threading import Thread
from typing import Any

import click
import segno

from agent_sessions import (
    agents,
    attend,
    attending,
    channel,
    db,
    display,
    index,
    limits,
    log,
    models,
    query,
    render,
    skill,
    topology,
    web,
)
from agent_sessions.agents import NotInstalled
from agent_sessions.forget import forget as forget_session
from agent_sessions.models import Block
from agent_sessions.render import Format, Renderer
from agent_sessions.resume import resume as resume_session
from agent_sessions.wormhole import WormholeUnavailable

EXIT_NO_RESULTS = 1
EXIT_USAGE = 2


class NoResults(Exception):
    """Nothing matched. Not an error, but worth its own exit code."""


def format_option(f):
    return click.option(
        "--format",
        "fmt",
        type=click.Choice([f.value for f in Format]),
        default=None,
        help="Output format. Defaults to agent when a coding agent or a pipe is detected.",
    )(
        click.option("--json", "as_json", is_flag=True, help="Shorthand for --format json.")(
            click.option(
                "-q", "--quiet", is_flag=True, help="Shorthand for --format quiet: ids only."
            )(f)
        )
    )


def renderer(fmt: str | None, as_json: bool, quiet: bool) -> Renderer:
    if sum(map(bool, (fmt, as_json, quiet))) > 1:
        raise click.UsageError("--format, --json and -q are mutually exclusive.")
    if as_json:
        return Renderer(Format.JSON)
    if quiet:
        return Renderer(Format.QUIET)
    return Renderer(Format(fmt) if fmt else render.detect())


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="""\b
Examples
  $ agent-sessions sync
  $ agent-sessions search "worktree relocation" -p wormhole
  $ agent-sessions ls -p temporal --since 2w
  $ agent-sessions show claude:7e90a7c6 --turns
  $ agent-sessions tree claude:7e90a7c6
  $ agent-sessions cat claude:7e90a7c6 --tools
  $ agent-sessions resume claude:7e90a7c6
  $ agent-sessions resume claude:7e90a7c6@9f3c1d20
  $ agent-sessions issue claude:7e90a7c6
  $ agent-sessions attend
""",
)
@click.version_option()
def main() -> None:
    """Find and resume the sessions you have had with coding agents.

    `sync` brings the index up to date with the transcript files on disk; every
    other command reads.
    """


@main.command()
@click.option(
    "--agent",
    "agents",
    multiple=True,
    type=click.Choice(sorted(index.SOURCES)),
    help="Only read this agent's transcripts. Repeatable. Defaults to all.",
)
@format_option
def sync(agents: tuple[str, ...], fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Bring the index into line with the transcript files. Safe to re-run.

    The only command that writes. Reads every transcript, so it takes about a
    second and there is nothing to keep incrementally correct.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    result = index.sync(conn, list(agents) or None)
    out.record(
        {
            "indexed": result.indexed,
            "nodes": result.nodes,
            "skipped": result.skipped,
            "duplicated": result.duplicated,
            "forgotten": result.forgotten,
        }
    )
    out.hint('Next: agent-sessions search "<words>" | agent-sessions ls -p <project> --since 2w')


@main.command()
@format_option
def status(fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Report index health: session counts, and whether a sync is due."""
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    total = conn.execute("SELECT count(*) FROM session").fetchone()[0]
    by_agent = conn.execute(
        "SELECT agent, count(*) AS n FROM session GROUP BY agent ORDER BY n DESC"
    ).fetchall()
    changed = index.stale(conn)
    out.record(
        {
            "db": str(db.DB_PATH),
            "sessions": total,
            **{row["agent"]: row["n"] for row in by_agent},
            "synced_at": _ago(db.synced_at(conn)),
            "changed_since": changed,
        }
    )


def filter_options(f):
    for option in reversed(
        [
            click.option("-p", "--project", help="Only this wormhole project, tasks included."),
            click.option(
                "--agent", type=click.Choice(sorted(index.SOURCES)), help="Only this agent."
            ),
            click.option("--since", help="Only agent-sessions touched within e.g. 36h, 10d, 2w."),
            click.option(
                "--min-context", help="Only agent-sessions whose context reached e.g. 500k."
            ),
            click.option(
                "-n", "--limit", type=int, default=20, show_default=True, help="How many at most."
            ),
        ]
    ):
        f = option(f)
    return f


def filters(project, agent, since, min_context) -> query.Filters:
    return query.Filters(project=project, agent=agent, since=since, min_context=min_context)


@main.command(
    epilog="""\b
Examples
  $ agent-sessions search compaction                    # case-insensitive, and stemmed:
                                                  # matches Compaction, compacted
  $ agent-sessions search "worktree relocation"         # both words, in any order
  $ agent-sessions search '"worktree relocation"'       # the exact phrase
  $ agent-sessions search 'nexus OR chasm'              # either
  $ agent-sessions search 'activity NOT workflow'       # one but not the other
  $ agent-sessions search 'reloc*'                      # prefix of the STEM: relocat*
                                                  # would find nothing
  $ agent-sessions search compaction -p temporal --since 2w
  $ agent-sessions search compaction --min-context 500k -n 5
  $ agent-sessions search compaction -q | head -1       # bare id, to pipe
"""
)
@click.argument("query_text", metavar="QUERY")
@filter_options
@format_option
def search(query_text, project, agent, since, min_context, limit, fmt, as_json, quiet) -> None:
    """Find agent-sessions whose text matches QUERY. Best match first, one per line.

    Matching is case-insensitive and stemmed, so `relocate` finds "relocating".
    QUERY is SQLite FTS5 syntax: bare words are ANDed, "quoted phrases" are
    literal, and OR / NOT work as expected. Ranking blends how well the text
    matches with how recent the session is.

    Stemming makes prefix search a trap: "relocating" is stored as `reloc`, so
    `reloc*` matches it and `relocat*` does not. Stemming usually covers what
    you wanted a prefix for anyway.

    Tool calls and their output are deliberately not indexed, so this searches
    what was said, not what was run. Use `agent-sessions cat --tools` for those.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    try:
        hits = query.search(conn, query_text, filters(project, agent, since, min_context), limit)
    except sqlite3.OperationalError as e:
        raise click.UsageError(f"bad query {query_text!r}: {e}") from e
    _report(
        out,
        [display.row(h) | {"snippet": display.snippet(h["snippet"], query_text)} for h in hits],
    )
    if hits:
        _next(out, hits[0]["id"], "show {id} --turns", "cat --tools", "tree", "resume")


@main.command(name="ls")
@filter_options
@click.option(
    "--sort",
    type=click.Choice(sorted(query.SORTS)),
    default="recent",
    show_default=True,
    help="Order by recency, peak context, or number of turns.",
)
@format_option
def list_sessions(project, agent, since, min_context, limit, sort, fmt, as_json, quiet) -> None:
    """List sessions, most recent first. No text matching; use `search` for that."""
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    found = query.recent(conn, filters(project, agent, since, min_context), sort, limit)
    _report(out, [display.row(f) for f in found])
    if found:
        _next(out, found[0]["id"], "show {id} --turns", "cat --tools", "tree", "resume")


@main.command()
@click.argument("id")
@click.option("--turns", "turns_only", is_flag=True, help="Just my turns, without the header.")
@format_option
def show(id: str, turns_only: bool, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Show one session and the history of my turns in it.

    Each turn carries the context size at that point, so it is visible where
    the session grew expensive and where compaction cut it back.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    session = _resolve(conn, id)
    said = query.turns(conn, session["id"])

    if as_json:
        out.record(dict(session) | {"turns": said})
        return
    if not turns_only:
        out.record(display.details(session))
        out.line("")
    out.table([display.turn(t) for t in said], quiet_key="at")
    _next(out, session["id"], "cat {id} --tools", "tree", "resume", "resume --fork")


@main.command()
@click.argument("id")
@format_option
def tree(id: str, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Show where a session branched, where it was compacted, and what forked off it.

    Long runs where nothing was decided collapse to one line each, so what is
    left is the shape. Compaction starts a new root, because the boundary record
    has no parent — there is no path back across it.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    session = _resolve(conn, id)
    shape = topology.of(conn, session)

    if as_json:
        out.record({"id": session["id"], **asdict(shape)})
        return
    out.line(f"{session['id']}  {session['project'] or '-'}  {session['title'] or ''}")
    if origin := shape.forked_from:
        out.line(f"  forked from {origin['parent']} at {display.point(origin['at_uuid'])}")
    for i, root in enumerate(shape.roots):
        _draw(out, root, prefix="", last=i == len(shape.roots) - 1)
    for fork in shape.forks:
        out.line(f"  fork → {fork['child']} at {display.point(fork['at_uuid'])}")
    _next(out, session["id"], "show {id} --turns", "cat --tools", "resume")


def _draw(out: Renderer, segment: topology.Segment, prefix: str, last: bool) -> None:
    """Each stretch is labelled with the point it ends at, which is what `resume` takes."""
    glyph = "└─ " if last else "├─ "
    out.line(f"{prefix}{glyph}{display.point(segment.end)}  {display.describe(segment)}")
    below = prefix + ("   " if last else "│  ")
    for i, child in enumerate(segment.children):
        _draw(out, child, below, i == len(segment.children) - 1)


@main.command(
    epilog="""\b
Examples
  $ agent-sessions cat claude:7e90a7c6 --tools         # the whole thing
  $ agent-sessions cat claude:7e90a7c6 --last          # only the turn just had
  $ agent-sessions cat claude:7e90a7c6@9f3c1d20        # only the turn around a point
"""
)
@click.argument("id", metavar="ID[@POINT]")
@click.option("--tools", is_flag=True, help="Include tool calls and their output.")
@click.option("--whole", is_flag=True, help="Include branches that were abandoned.")
@click.option("--last", is_flag=True, help="Only the most recent turn.")
def cat(id: str, tools: bool, whole: bool, last: bool) -> None:
    """Print a session as markdown, read from the transcript itself.

    Not from the index, which holds no tool output: `--tools` is the only way
    to see what was actually run.

    A turn is what I asked and everything that followed. `--last` gives the one
    just had; `@<point>` gives the one that point falls in, taking the points
    `show --turns` and `tree` print.
    """
    conn = db.connect()
    session, at = _resolve_point(conn, id)
    if at and last:
        raise click.UsageError("--last and a point name different turns. Ask for one of them.")
    source = index.SOURCES[session["agent"]]
    path = Path(session["path"])
    if not path.exists():
        raise click.UsageError(f"{path} is gone. Run `agent-sessions sync`.")
    if not (at or last):
        _write(source.render(path, tools=tools, whole=whole))
        return
    _write(source.render_blocks(_one_turn(source.blocks(path, whole=whole), at, last), tools=tools))


def _one_turn(conversation: list[Block], at: str, last: bool) -> list[Block]:
    said = models.turns(conversation)
    if not said:
        raise click.UsageError("nothing was asked in this session, so it holds no turn.")
    turn = said[-1] if last else models.turn_at(conversation, at)
    if turn is None:
        raise click.UsageError(
            f"{display.point(at)} is in no turn of this session."
            " Try a point from `agent-sessions show --turns`."
        )
    return turn.whole


def _write(chunks: Iterator[str]) -> None:
    for chunk in chunks:
        sys.stdout.write(chunk)


@main.command(
    epilog="""\b
Examples
  $ agent-sessions resume claude:7e90a7c6             # where it was left
  $ agent-sessions resume claude:7e90a7c6 --fork      # branch, leaving it as it is
  $ agent-sessions resume claude:7e90a7c6@9f3c1d20    # from a point inside it
  $ agent-sessions resume claude:7e90a7c6 --remote    # to carry on from a phone
"""
)
@click.argument("id", metavar="ID[@POINT]")
@click.option("--fork", is_flag=True, help="Branch into a new session, leaving this one as it is.")
@click.option("--remote", is_flag=True, help="Hand it to the agent's remote control, for a phone.")
@format_option
def resume(id: str, fork: bool, remote: bool, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Pick a session back up, in the worktree it belongs to.

    A session that is already running is not started again: you are taken to
    the pane it is sitting in, whoever started it. Otherwise wormhole splits a
    pane in the project's window and resumes it there.

    `<id>@<point>` picks it up from a point inside it rather than from where it
    was left — before a compaction, or before a turn that went wrong. `tree`
    and `show --turns` print the points. Resuming at one is always a fork: what
    starts is a new session ending there, and this one is left as it is.

    `--remote` starts it with the agent's own remote control on, so it can be
    carried on from a phone. The agent can already do that for the session in
    front of you; picking up an older one is the part it cannot do, because its
    session picker runs only in the terminal.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    session, at = _resolve_point(conn, id)
    resumed = resume_session(session, fork=fork, at=at, remote=remote)
    out.record(asdict(resumed))
    if resumed.was_running and not fork:
        out.hint(
            f"Already running ({resumed.was_running}); focused its pane rather than starting again."
        )
    if remote and resumed.was_running and not fork:
        out.hint("Remote control cannot be put on it from out here; type /rc in it.")
    if resumed.remote_home:
        out.hint(f"Carry it on at {resumed.remote_home}, or in the Claude app.")
    if resumed.at:
        out.hint(f"{resumed.resumed_as} is new; run `agent-sessions sync` to index it.")


@main.command()
@click.argument("id")
@click.argument("title")
@format_option
def rename(id: str, title: str, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Give a session a title of my own, in place of the one it was given.

    Written where the agent keeps its own titles, so it survives a sync and
    shows in the agent's own UI too.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    session = _resolve(conn, id)
    index.SOURCES[session["agent"]].retitle(Path(session["path"]), title)
    db.set_title(conn, session["id"], title)
    conn.commit()
    out.record({"id": session["id"], "title": title})


@main.command(
    epilog="""\b
Examples
  $ agent-sessions forget claude:7e90a7c6       # by the id Claude prints as it exits
  $ agent-sessions forget 7e90a7c6              # or any unambiguous prefix of it
"""
)
@click.argument("id")
@format_option
def forget(id: str, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Take a session out of the index, and its transcript out of the agent's reach.

    Not a delete: the transcript is moved to `~/.agent-sessions/forgotten`, and
    moving it back is all it would take. Its prompts stop being offered at the
    agent's input line, which is the other half of being gone.

    A session still running cannot be forgotten — its agent has the file open —
    so this is for after quitting one, with the id it printed on its way out.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    out.record(asdict(forget_session(conn, _name(conn, id))) | {"forgotten": True})


def _name(conn: sqlite3.Connection, id: str) -> str:
    """The session that id names, indexed or not.

    The id to hand this is the one Claude prints as it exits, and that session
    has not been synced: it ended a moment ago. So an id naming a transcript
    that is there is taken as it is; a prefix still has to be looked up.
    """
    agent, _, native_id = id.rpartition(":")
    for name, source in index.SOURCES.items():
        if agent in ("", name) and source.locate(native_id):
            return f"{name}:{native_id}"
    return _resolve(conn, id)["id"]


@main.command(
    epilog="""\b
Examples
  $ agent-sessions issue claude:7e90a7c6      # reach it from a phone
"""
)
@click.argument("id")
@format_option
def issue(id: str, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Put a session on the control channel, so it can be carried on from anywhere.

    An issue in a private repo, whose body says which session it is. A comment
    of mine on it is a prompt; `attend` notices it, runs the turn here, and
    posts what came back. Nothing listens on this machine for any of it.

    Asking twice gives the same issue: which session an issue is for is written
    in the issue, not in the index.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    session = _resolve(conn, id)
    opened = attend.issue_for(channel.Channel(), session)
    out.record({"id": session["id"], "issue": opened.number, "url": opened.url})
    out.hint(f"Comment on {opened.url} to carry it on. `agent-sessions attend` must be running.")


@main.command(name="attend")
@click.option(
    "--interval",
    default=attend.INTERVAL,
    show_default=True,
    help="Seconds between polls. A poll that finds nothing costs no rate limit.",
)
@click.option("--once", "single", is_flag=True, help="One pass, rather than staying up.")
@click.option("--verbose", is_flag=True, help="Say something on every poll, not just events.")
@click.option("--reset", is_flag=True, help="Clear the breaker and say what tripped it.")
def attend_channel(interval: float, single: bool, verbose: bool, reset: bool) -> None:
    """Answer prompts left on the control channel, until interrupted.

    Polls for comments, runs each turn in the worktree its session belongs to,
    and posts the result back. A turn runs under my own settings, so a comment
    on the issue is allowed whatever I am allowed.
    """
    if reset:
        was = limits.reset()
        click.echo(f"Cleared: {was}" if was else "It was not tripped.")
        return
    log.VERBOSE = verbose
    conn = db.connect()
    control = channel.Channel()
    log.say(f"attending {control.repo}, every {interval:g}s")
    if single:
        attend.once(control, conn)
        return
    with suppress(KeyboardInterrupt):
        _attending(lambda: attend.loop(control, conn, interval))


@main.command()
@click.option("--port", default=web.PORT, show_default=True, help="Which port to listen on.")
@click.option("--host", default=web.HOST, show_default=True, help="Which address to bind.")
@click.option("--lan", is_flag=True, help="Serve to the rest of the network, for a phone.")
@click.option("--attend", "attending_too", is_flag=True, help="Answer the control channel too.")
@click.option("--verbose", is_flag=True, help="Say something on every poll, not just events.")
@click.option("--open/--no-open", "open_browser", default=True, help="Open a browser at it.")
def serve(
    port: int, host: str, lan: bool, attending_too: bool, verbose: bool, open_browser: bool
) -> None:
    """Serve the index in a browser, where a link is enough to resume.

    `/resume/<id>` resumes on a GET, so that URL is clickable from anywhere a URL
    can be clicked — a note, a chat message, an agent's output — not only from
    this UI. Runs until interrupted.

    `--lan` serves it to the rest of the network, which is how a phone gets to
    it, and prints a QR code so nobody has to read an address out. Everything
    on that network can then reach every conversation you have had, and resume
    one, so it is asked for rather than assumed.

    `--attend` answers the control channel alongside, so leaving this running
    is the whole of the setup. Serving is reading and attending runs an agent,
    which is why one does not imply the other.
    """
    if lan:
        host = web.EVERY_INTERFACE
    local = f"http://{web.HOST}:{port}/"
    print(f"agent-sessions at {local}", file=sys.stderr)
    if lan:
        _announce(web.reachable(port))
    log.VERBOSE = verbose
    server = _bound(host, port)
    if attending_too:
        _attend_beside(server)
        return
    if open_browser:
        webbrowser.open(local)
    with suppress(KeyboardInterrupt), server:
        server.serve_forever()


def _bound(host: str, port: int) -> Any:
    """The socket, before anything else is started. Say who has it if it is taken."""
    try:
        return web.listener(host, port)
    except OSError as e:
        raise click.UsageError(
            f"cannot listen on {host}:{port} ({e.strerror})."
            " Another agent-sessions serve is the usual reason."
        ) from e


def _attend_beside(server: Any) -> None:
    """The channel in this thread, the pages in another. Both stop on one interrupt.

    This way round because a failure attending is one I should see, and the
    socket is already bound by the time the thread gets it.
    """
    control = channel.Channel()
    log.say(f"attending {control.repo}, every {attend.INTERVAL:g}s")
    Thread(target=server.serve_forever, daemon=True).start()
    with suppress(KeyboardInterrupt), server:
        _attending(lambda: attend.loop(control, db.connect()))


def _attending(run: Callable[[], None]) -> None:
    """Only one loop may attend, and a second should say so rather than fight."""
    try:
        run()
    except attending.AlreadyAttending as e:
        raise click.UsageError(str(e)) from e
    except channel.NotAuthorized as e:
        raise click.UsageError(
            f"{e} Check AGENT_WORK_APP_ID and AGENT_WORK_APP_KEY, or `gh auth status`."
        ) from e


def _announce(urls: list[str]) -> None:
    """A QR of the first address, and the rest in writing for a phone that cannot scan."""
    for url in urls:
        print(f"           and at {url}", file=sys.stderr)
    # Compact, because a cell per module is twice as tall as it is wide and the
    # code comes out too big for a terminal to show at once.
    segno.make(urls[0]).terminal(out=sys.stderr, border=2, compact=True)


@main.command(
    epilog="""\b
Examples
  $ agent-sessions agent                                    # pi, knowing this tool
  $ agent-sessions agent --with qwen
  $ agent-sessions agent --model anthropic/claude-opus-5
  $ agent-sessions agent "pick up the compaction work"      # your own first task
"""
)
@click.option(
    "--with",
    "program",
    type=click.Choice(sorted(agents.DEFAULT_MODELS)),
    default="pi",
    show_default=True,
    help="Which coding agent to start.",
)
@click.option("--model", help="Which model it should run, named as that agent names it.")
@click.argument("prompt", required=False)
def agent(program: str, model: str | None, prompt: str | None) -> None:
    """Start a coding agent that already knows this tool.

    The whole skill goes in the opening message, so it is in context from the
    first turn rather than merely discoverable. Neither agent can pre-fill its
    input without submitting it, so the message carries a task too: PROMPT if
    given, otherwise a look at your recent sessions.

    This process is replaced by the agent, in the current directory.
    """
    agents.start(program, model or "", prompt or "")


@main.group(name="skills")
def skills_group() -> None:
    """Manage the agent-sessions skill, so an agent knows this tool from turn one."""


@skills_group.command(name="add")
@format_option
def skills_add(fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Install the skill into ~/.agents/skills. Symlink from elsewhere if wanted."""
    out = renderer(fmt, as_json, quiet)
    path = skill.install()
    out.record({"path": str(path), "bytes": path.stat().st_size})


@skills_group.command(name="preview")
def skills_preview() -> None:
    """Print the skill without installing it."""
    sys.stdout.write(skill.generate())


def _resolve(conn: sqlite3.Connection, id: str) -> dict:
    session = query.get(conn, id)
    if session is None:
        raise click.UsageError(
            f"no single session matches {id!r}. Try `agent-sessions search` or a longer prefix."
        )
    return session


def _resolve_point(conn: sqlite3.Connection, id: str) -> tuple[dict, str]:
    """`<id>`, or `<id>@<point>` naming somewhere inside it. Both resolve by prefix."""
    id, at = query.split_point(id)
    session = _resolve(conn, id)
    if not at:
        return session, ""
    point = query.node(conn, session["id"], at)
    if point is None:
        raise click.UsageError(
            f"no single point in {session['id']} matches {at!r}."
            " Try `agent-sessions tree` or a longer prefix."
        )
    return session, point["uuid"]


def _report(out: Renderer, rows: list[dict]) -> None:
    out.table(rows)
    if not rows:
        raise NoResults()


def _next(out: Renderer, id: str, ready: str, *others: str) -> None:
    """One command ready to run, then the rest of the vocabulary.

    A worked example saves an agent a call on --help. Spelling the id out
    three times would not: one runnable command and the other verbs is
    enough to act on.
    """
    out.hint(f"Next: agent-sessions {ready.format(id=id)}   (also: {', '.join(others)})")


def _ago(when: int | None) -> str:
    if when is None:
        return "never"
    seconds = int(time.time()) - when
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds // size}{unit} ago"
    return "just now"


def run(argv: list[str] | None = None) -> int:
    """The entry point, and the whole of the exit-code contract."""
    try:
        main.main(args=argv, standalone_mode=False)
    except NoResults:
        return EXIT_NO_RESULTS
    except (click.UsageError, ValueError) as e:
        message = e.format_message() if isinstance(e, click.UsageError) else str(e)
        print(f"Error: {message}", file=sys.stderr)
        return EXIT_USAGE
    except click.exceptions.Abort:
        return EXIT_USAGE
    except (NotInstalled, WormholeUnavailable) as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_USAGE
    return 0


if __name__ == "__main__":
    sys.exit(run())
