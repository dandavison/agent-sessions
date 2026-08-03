"""The agent-sessions command line."""

import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

import click

from agent_sessions import agents, db, index, query, render, skill, topology, wormhole
from agent_sessions.agents import NotInstalled
from agent_sessions.render import Format, Renderer
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
    _report(out, [_row(h) | {"snippet": _snippet(h["snippet"], query_text)} for h in hits])
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
    _report(out, [_row(f) for f in found])
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
        out.record(_details(session))
        out.line("")
    out.table([_turn(t) for t in said], quiet_key="uuid")
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
        out.line(f"  forked from {origin['parent']} at {_short(origin['at_uuid'])}")
    for i, root in enumerate(shape.roots):
        _draw(out, root, prefix="", last=i == len(shape.roots) - 1)
    for fork in shape.forks:
        out.line(f"  fork → {fork['child']} at {_short(fork['at_uuid'])}")
    _next(out, session["id"], "show {id} --turns", "cat --tools", "resume")


def _draw(out: Renderer, segment: topology.Segment, prefix: str, last: bool) -> None:
    out.line(prefix + ("└─ " if last else "├─ ") + _describe(segment))
    below = prefix + ("   " if last else "│  ")
    for i, child in enumerate(segment.children):
        _draw(out, child, below, i == len(segment.children) - 1)


def _describe(segment: topology.Segment) -> str:
    if c := segment.compaction:
        dropped = (c["pre_tokens"] or 0) - (c["post_tokens"] or 0)
        return (
            f"compacted ({c['trigger']}) {_tokens(c['pre_tokens'])} → {_tokens(c['post_tokens'])},"
            f" dropped {_tokens(dropped)}, {c['preserved_count']} kept"
        )
    parts = [
        _date(segment.started_at, with_time=True),
        _plural(segment.turns, "turn") if segment.turns else _plural(segment.messages, "message"),
        _tokens(segment.context_tokens),
        "(abandoned)" if segment.abandoned else "",
    ]
    return "  ".join(p for p in parts if p)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _short(uuid: str | None) -> str:
    return (uuid or "")[:8]


@main.command()
@click.argument("id")
@click.option("--tools", is_flag=True, help="Include tool calls and their output.")
@click.option("--whole", is_flag=True, help="Include branches that were abandoned.")
def cat(id: str, tools: bool, whole: bool) -> None:
    """Print a session as markdown, read from the transcript itself.

    Not from the index, which holds no tool output: `--tools` is the only way
    to see what was actually run.
    """
    conn = db.connect()
    session = _resolve(conn, id)
    source = index.SOURCES[session["agent"]]
    path = Path(session["path"])
    if not path.exists():
        raise click.UsageError(f"{path} is gone. Run `agent-sessions sync`.")
    for chunk in source.render(path, tools=tools, whole=whole):
        sys.stdout.write(chunk)


@main.command()
@click.argument("id")
@click.option("--fork", is_flag=True, help="Branch into a new session, leaving this one as it is.")
@format_option
def resume(id: str, fork: bool, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Pick a session back up, in the worktree it belongs to.

    A session that is already running is not started again: you are taken to
    the pane it is sitting in, whoever started it. Otherwise wormhole splits a
    pane in the project's window and resumes it there.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    session = _resolve(conn, id)
    if not session["project"]:
        raise click.UsageError(
            f"{session['id']} has no project: its cwd was {session['cwd']}."
            " There is nowhere to resume it."
        )

    running = index.SOURCES[session["agent"]].live().get(session["native_id"])
    wormhole.resume(
        session["project"],
        session["native_id"],
        fork=fork,
        pid=running.pid if running else None,
    )
    out.record(
        {
            "id": session["id"],
            "project": session["project"],
            "forked": fork,
            "was_running": running.status if running else "",
        }
    )
    if running and not fork:
        out.hint(
            f"Already running ({running.status}); focused its pane rather than starting again."
        )


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


def _row(s: dict) -> dict:
    return {
        "id": s["id"],
        "project": s["project"],
        "when": _date(s["ended_at"]),
        "turns": s["n_user_turns"],
        "context": _tokens(s["context_tokens"]),
        "title": s["title"],
    }


def _details(s: dict) -> dict:
    return {
        "id": s["id"],
        "title": s["title"],
        "project": s["project"],
        "branch": s["git_branch"],
        "cwd": s["cwd"],
        "model": s["model"],
        "started": _date(s["started_at"]),
        "ended": _date(s["ended_at"]),
        "turns": s["n_user_turns"],
        "messages": s["n_messages"],
        "context": _tokens(s["context_tokens"]),
        "output": _tokens(s["output_tokens"]),
        "dropped": _tokens(s["dropped_tokens"]),
    }


def _turn(t: dict) -> dict:
    return {
        "when": _date(t["ts"], with_time=True),
        "role": t["role"],
        "context": _tokens(t["context_tokens"]),
        "text": t["text"],
    }


def _snippet(text: str, query_text: str) -> str:
    """The matching node, trimmed to the neighbourhood of the first matching word."""
    flat = " ".join(text.split())
    words = [w.strip('"*').lower() for w in query_text.split() if w.isalnum()]
    lowered = flat.lower()
    at = next((i for w in words if (i := lowered.find(w)) >= 0), 0)
    start = max(0, at - 40)
    return ("…" if start else "") + flat[start : start + 200]


def _date(when: int | None, with_time: bool = False) -> str:
    if not when:
        return ""
    shape = "%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d"
    return time.strftime(shape, time.localtime(when))


def _tokens(n: int | None) -> str:
    if not n:
        return ""
    return f"{n / 1_000_000:.1f}m" if n >= 1_000_000 else f"{n // 1000}k" if n >= 1000 else str(n)


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
