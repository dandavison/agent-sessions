"""The skill: what an agent needs to know before its first call.

Generated from the command tree rather than written by hand, so it cannot drift
from the CLI it describes.
"""

from pathlib import Path
from textwrap import dedent

import click

SKILLS_DIR = Path.home() / ".agents" / "skills" / "senderos"

FRONTMATTER = """---
name: senderos
description: >-
  Find and resume past work done with coding agents. Use when the user refers to
  something they did before — "the conversation where we…", "what did I decide
  about…", "pick up where I left off on…" — or asks how much context a piece of
  work used, where it branched, or what they said in it.
---
"""

PREAMBLE = """
# senderos

A *sendero* is one unit of work done with a coding agent: durable, branching,
resumable. `senderos` indexes them and finds them again.

Output adapts to the caller: you will get TSV with nothing truncated, and full
ids and timestamps. Add `--json` for a structured payload, `-q` for bare ids to
pipe. Hints and errors go to stderr; only data goes to stdout.

Exit codes: 0 found something, 1 found nothing, 2 you called it wrong.

## Typical use

    senderos search "worktree relocation" -p wormhole   # find it
    senderos show <id> --turns                          # read what was said
    senderos resume <id>                                # pick it back up

Ids round-trip: whatever `search` prints is accepted by every other command,
and any unambiguous prefix will do — `senderos show 7daeb` is enough.

## Worth knowing

- The index is a cache over transcript files. It is not updated automatically;
  if `changed_since` is non-zero in `senderos status`, run `senderos sync`.
- Tool calls and their output are not indexed, so `search` matches what was
  said, not what was run. Use `senderos cat <id> --tools` to see commands.
- Context size is the peak the sendero reached, not a sum of its turns.

## Commands
"""


def generate(command: click.Group | None = None) -> str:
    """One line per command, plus the flags that are not self-evident."""
    from senderos.cli import main

    command = command if command is not None else main
    lines = [FRONTMATTER.strip(), PREAMBLE.rstrip(), ""]
    for name, sub in sorted(command.commands.items()):
        lines.append(f"### `{' '.join(['senderos', name, *_arguments(sub)])}`")
        lines.append("")
        lines.append(_summary(sub))
        if options := _options(sub):
            lines.append("")
            lines += options
        if examples := _examples(sub):
            lines.append("")
            lines += examples
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _examples(command: click.Command) -> list[str]:
    """Worked examples, which an agent matches against faster than it reads prose.

    Indentation is preserved, not stripped: a comment continued onto the next
    line only reads as a continuation if it stays under the one it follows.
    """
    if not command.epilog:
        return []
    body = command.epilog.replace("\b\n", "").replace("Examples\n", "")
    return ["```", *dedent(body).strip("\n").splitlines(), "```"]


def _arguments(command: click.Command) -> list[str]:
    ctx = click.Context(command)
    return [p.make_metavar(ctx) for p in command.params if isinstance(p, click.Argument)]


def _summary(command: click.Command) -> str:
    return (command.help or "").strip().split("\n\n")[0].replace("\n    ", " ")


# Flags every command carries; repeating them per command would only cost context.
UNIVERSAL = {"help", "fmt", "as_json", "quiet"}


def _options(command: click.Command) -> list[str]:
    lines = []
    for param in command.params:
        if not isinstance(param, click.Option) or param.name in UNIVERSAL:
            continue
        flags = ", ".join(param.opts)
        lines.append(f"- `{flags}` — {param.help or ''}".rstrip(" —"))
    return lines


def install(directory: Path = SKILLS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(generate())
    return path
