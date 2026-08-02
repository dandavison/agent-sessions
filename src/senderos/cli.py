"""The senderos command line."""

import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

import click

from senderos import db, index, query, render, skill, topology, wormhole
from senderos.render import Format, Renderer
from senderos.wormhole import WormholeUnavailable

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
  $ senderos sync
  $ senderos search "worktree relocation" -p wormhole
  $ senderos ls -p temporal --since 2w
  $ senderos show claude:7e90a7c6 --turns
  $ senderos tree claude:7e90a7c6
  $ senderos cat claude:7e90a7c6 --tools
  $ senderos resume claude:7e90a7c6
""",
)
@click.version_option()
def main() -> None:
    """Index and resume the work you have done with coding agents.

    A sendero is one unit of that work: a durable, branching, resumable piece of
    it. `sync` brings the index up to date with the transcript files on disk;
    every other command reads.
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
    out.hint("Use `senderos search <query>` to find one.")


@main.command()
@format_option
def status(fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Report index health: sendero counts, and whether a sync is due."""
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    total = conn.execute("SELECT count(*) FROM sendero").fetchone()[0]
    by_agent = conn.execute(
        "SELECT agent, count(*) AS n FROM sendero GROUP BY agent ORDER BY n DESC"
    ).fetchall()
    changed = index.stale(conn)
    out.record(
        {
            "db": str(db.DB_PATH),
            "senderos": total,
            **{row["agent"]: row["n"] for row in by_agent},
            "synced_at": _ago(db.synced_at(conn)),
            "changed_since": changed,
        }
    )
    if changed:
        out.hint(f"{changed} transcripts have changed. Run `senderos sync`.")


def filter_options(f):
    for option in reversed(
        [
            click.option("-p", "--project", help="Only this wormhole project, tasks included."),
            click.option(
                "--agent", type=click.Choice(sorted(index.SOURCES)), help="Only this agent."
            ),
            click.option("--since", help="Only senderos touched within e.g. 36h, 10d, 2w."),
            click.option("--min-context", help="Only senderos whose context reached e.g. 500k."),
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
  $ senderos search compaction                    # case-insensitive, and stemmed:
                                                  # matches Compaction, compacted
  $ senderos search "worktree relocation"         # both words, in any order
  $ senderos search '"worktree relocation"'       # the exact phrase
  $ senderos search 'nexus OR chasm'              # either
  $ senderos search 'activity NOT workflow'       # one but not the other
  $ senderos search 'reloc*'                      # prefix of the STEM: relocat*
                                                  # would find nothing
  $ senderos search compaction -p temporal --since 2w
  $ senderos search compaction --min-context 500k -n 5
  $ senderos search compaction -q | head -1       # bare id, to pipe
"""
)
@click.argument("query_text", metavar="QUERY")
@filter_options
@format_option
def search(query_text, project, agent, since, min_context, limit, fmt, as_json, quiet) -> None:
    """Find senderos whose text matches QUERY. Best match first, one per line.

    Matching is case-insensitive and stemmed, so `relocate` finds "relocating".
    QUERY is SQLite FTS5 syntax: bare words are ANDed, "quoted phrases" are
    literal, and OR / NOT work as expected. Ranking blends how well the text
    matches with how recent the sendero is.

    Stemming makes prefix search a trap: "relocating" is stored as `reloc`, so
    `reloc*` matches it and `relocat*` does not. Stemming usually covers what
    you wanted a prefix for anyway.

    Tool calls and their output are deliberately not indexed, so this searches
    what was said, not what was run. Use `senderos cat --tools` for those.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    try:
        hits = query.search(conn, query_text, filters(project, agent, since, min_context), limit)
    except sqlite3.OperationalError as e:
        raise click.UsageError(f"bad query {query_text!r}: {e}") from e
    _report(out, conn, [_row(h) | {"snippet": _snippet(h["snippet"], query_text)} for h in hits])
    if hits:
        out.hint(f"`senderos show {hits[0]['id']}` for the whole of the top hit.")


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
def list_senderos(project, agent, since, min_context, limit, sort, fmt, as_json, quiet) -> None:
    """List senderos, most recent first. No text matching; use `search` for that."""
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    found = query.recent(conn, filters(project, agent, since, min_context), sort, limit)
    _report(out, conn, [_row(f) for f in found])


@main.command()
@click.argument("id")
@click.option("--turns", "turns_only", is_flag=True, help="Just my turns, without the header.")
@format_option
def show(id: str, turns_only: bool, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Show one sendero and the history of my turns in it.

    Each turn carries the context size at that point, so it is visible where
    the sendero grew expensive and where compaction cut it back.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    sendero = _resolve(conn, id)
    said = query.turns(conn, sendero["id"])

    if as_json:
        out.record(dict(sendero) | {"turns": said})
        return
    if not turns_only:
        out.record(_details(sendero))
        out.line("")
    out.table([_turn(t) for t in said], quiet_key="uuid")
    out.hint(f"`senderos resume {sendero['id']}` to pick it up.")


@main.command()
@click.argument("id")
@format_option
def tree(id: str, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Show where a sendero branched, where it was compacted, and what forked off it.

    Long runs where nothing was decided collapse to one line each, so what is
    left is the shape. Compaction starts a new root, because the boundary record
    has no parent — there is no path back across it.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    sendero = _resolve(conn, id)
    shape = topology.of(conn, sendero)

    if as_json:
        out.record({"id": sendero["id"], **asdict(shape)})
        return
    out.line(f"{sendero['id']}  {sendero['project'] or '-'}  {sendero['title'] or ''}")
    if origin := shape.forked_from:
        out.line(f"  forked from {origin['parent']} at {_short(origin['at_uuid'])}")
    for i, root in enumerate(shape.roots):
        _draw(out, root, prefix="", last=i == len(shape.roots) - 1)
    for fork in shape.forks:
        out.line(f"  fork → {fork['child']} at {_short(fork['at_uuid'])}")


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
    """Print a sendero as markdown, read from the transcript itself.

    Not from the index, which holds no tool output: `--tools` is the only way
    to see what was actually run.
    """
    conn = db.connect()
    sendero = _resolve(conn, id)
    source = index.SOURCES[sendero["agent"]]
    path = Path(sendero["path"])
    if not path.exists():
        raise click.UsageError(f"{path} is gone. Run `senderos sync`.")
    for chunk in source.render(path, tools=tools, whole=whole):
        sys.stdout.write(chunk)


@main.command()
@click.argument("id")
@click.option("--fork", is_flag=True, help="Branch into a new session, leaving this one as it is.")
@format_option
def resume(id: str, fork: bool, fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Pick a sendero back up, in the worktree it belongs to.

    A session that is already running is not started again: you are taken to
    the pane it is sitting in, whoever started it. Otherwise wormhole splits a
    pane in the project's window and resumes it there.
    """
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    sendero = _resolve(conn, id)
    if not sendero["project"]:
        raise click.UsageError(
            f"{sendero['id']} has no project: its cwd was {sendero['cwd']}."
            " There is nowhere to resume it."
        )

    running = index.SOURCES[sendero["agent"]].live().get(sendero["native_id"])
    wormhole.resume(
        sendero["project"],
        sendero["native_id"],
        fork=fork,
        pid=running.pid if running else None,
    )
    out.record(
        {
            "id": sendero["id"],
            "project": sendero["project"],
            "forked": fork,
            "was_running": running.status if running else "",
        }
    )
    if running and not fork:
        out.hint(f"It is already running ({running.status}); focusing its pane.")


@main.group(name="skills")
def skills_group() -> None:
    """Manage the senderos skill, so an agent knows this tool from turn one."""


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
    sendero = query.get(conn, id)
    if sendero is None:
        raise click.UsageError(
            f"no single sendero matches {id!r}. Try `senderos search` or a longer prefix."
        )
    return sendero


def _report(out: Renderer, conn: sqlite3.Connection, rows: list[dict]) -> None:
    out.table(rows)
    if not rows:
        raise NoResults()
    if changed := index.stale(conn):
        out.hint(f"{changed} transcripts have changed since the last sync. Run `senderos sync`.")


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


def run() -> int:
    try:
        main.main(standalone_mode=False)
    except NoResults:
        return EXIT_NO_RESULTS
    except (click.UsageError, ValueError) as e:
        message = e.format_message() if isinstance(e, click.UsageError) else str(e)
        print(f"Error: {message}", file=sys.stderr)
        return EXIT_USAGE
    except click.exceptions.Abort:
        return EXIT_USAGE
    except WormholeUnavailable as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_USAGE
    return 0


if __name__ == "__main__":
    sys.exit(run())
