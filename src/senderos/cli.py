"""The senderos command line."""

import sys

import click

from senderos import db, render
from senderos.render import Format, Renderer
from senderos.wormhole import WormholeUnavailable

EXIT_NO_RESULTS = 1
EXIT_USAGE = 2
EXIT_NO_INDEX = 3


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
@format_option
def status(fmt: str | None, as_json: bool, quiet: bool) -> None:
    """Report index health: sendero counts, and whether a sync is due."""
    out = renderer(fmt, as_json, quiet)
    conn = db.connect()
    total = conn.execute("SELECT count(*) FROM sendero").fetchone()[0]
    by_agent = conn.execute(
        "SELECT agent, count(*) AS n FROM sendero GROUP BY agent ORDER BY n DESC"
    ).fetchall()
    out.record(
        {
            "db": str(db.DB_PATH),
            "senderos": total,
            **{row["agent"]: row["n"] for row in by_agent},
        }
    )
    if not total:
        out.hint("Run `senderos sync` to build the index.")


def run() -> int:
    try:
        main.main(standalone_mode=False)
    except click.UsageError as e:
        print(f"Error: {e.format_message()}", file=sys.stderr)
        return EXIT_USAGE
    except click.exceptions.Abort:
        return EXIT_USAGE
    except WormholeUnavailable as e:
        print(f"Error: {e}", file=sys.stderr)
        return EXIT_USAGE
    return 0


if __name__ == "__main__":
    sys.exit(run())
