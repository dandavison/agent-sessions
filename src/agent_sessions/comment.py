"""A turn as a GitHub comment, and a session as a GitHub issue.

The park is not on the local network, so the channel is inverted: nothing
listens on this machine, and the laptop reaches out to GitHub instead. What
that buys is a reader I do not have to write — GitHub renders markdown with
syntax highlighting, folds `<details>`, notifies a phone, and has an app.

So a turn is laid out for that reader. What the agent said goes at the top
level, because a notification opens on it and an answer should not need a tap;
everything it ran folds away beneath. The remaining view — mine alone, with no
agent output at all — is not something a timeline can do, since GitHub has no
collapse-all, so the issue body carries a running list of my prompts instead.

Everything on the way out goes through `redact`. Tool output is read off this
machine and holds what a commit never would.
"""

import re
from pathlib import Path
from typing import Any

import orjson

from agent_sessions.models import Block, Boundary, Ran, Said

# What GitHub accepts in one comment. Truncation is not a nicety: a test run
# clears this on its own and the post would simply fail.
LIMIT = 65_536

# Room for the footer and the closing tags that truncation must not cut into.
MARGIN = 400

REDACTED = "«redacted»"

# High-confidence shapes only. This is not a guarantee that nothing leaks — it
# cannot be — it is the part that can be done without guessing at prose.
SECRETS = re.compile(
    r"""
    sk-ant-[A-Za-z0-9_-]{20,}      # Anthropic
  | sk-[A-Za-z0-9]{32,}            # OpenAI
  | gh[pousr]_[A-Za-z0-9]{20,}     # GitHub
  | github_pat_[A-Za-z0-9_]{20,}
  | xox[abprs]-[A-Za-z0-9-]{10,}   # Slack
  | AKIA[0-9A-Z]{16}               # AWS
  | ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}  # JWT
    """,
    re.VERBOSE,
)

# The fence language for a tool that has one of its own. Anything else is shown
# as its input, which is JSON.
FENCES = {"Bash": "sh", "BashOutput": "sh"}

# Enough of the extensions in this corpus to be worth the highlighting.
LANGUAGES = {
    ".py": "python",
    ".rs": "rust",
    ".go": "go",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".java": "java",
    ".sh": "sh",
    ".zsh": "sh",
    ".sql": "sql",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".proto": "proto",
}

# The file a tool is about, whichever of these it calls it.
PATH_KEYS = ("file_path", "path", "notebook_path")


def render(blocks: list[Block], summary: dict[str, Any] | None = None) -> str:
    """A turn as the comment to post for it: a formatter over the one parser.

    The same blocks the terminal and the pages are shown, so a conversation
    cannot read one way here and another way there.
    """
    parts: list[str] = []
    for block in blocks:
        match block:
            case Said(role="assistant", text=text):
                parts.append(text.strip())
            case Ran():
                parts.append(_tool(block))
            case Boundary():
                parts.append(
                    f"<sub>Compacted ({block.trigger}): "
                    f"{block.pre_tokens:,} → {block.post_tokens:,} tokens</sub>"
                )
            case _:
                pass
    parts.append(_footer(summary))
    return _fit(redact("\n\n".join(p for p in parts if p)))


def body(session: dict[str, Any], prompts: list[str]) -> str:
    """The issue body: what this session is, and everything I have said in it.

    A table rather than YAML because it has to read on a phone as well as parse
    here, and it is the first thing on the page above the whole conversation.
    """
    rows = "\n".join(f"| {k} | {v} |" for k, v in _frontmatter(session).items() if v)
    said = "\n".join(f"{i}. {_quote(p)}" for i, p in enumerate(prompts, 1))
    return f"| | |\n|---|---|\n{rows}\n\n### What I have said\n\n{said or '*Nothing yet.*'}\n"


def session_id(body_text: str) -> str:
    """The session an issue is for, read back off the table it was written into."""
    found = re.search(r"^\|\s*session\s*\|\s*(\S+)\s*\|", body_text, re.MULTILINE)
    return found.group(1) if found else ""


def redact(text: str) -> str:
    return SECRETS.sub(REDACTED, text)


def _frontmatter(session: dict[str, Any]) -> dict[str, str]:
    return {
        "session": str(session.get("id", "")),
        "project": str(session.get("project") or ""),
        "cwd": str(session.get("cwd") or ""),
    }


def _quote(prompt: str) -> str:
    """First line only. This is an index of what I said, not a second copy of it."""
    first = prompt.strip().splitlines()[0] if prompt.strip() else ""
    return first if len(first) <= 120 else first[:117] + "…"


def _tool(call: Ran) -> str:
    """A tool call folded away, saying enough in the summary to not need opening."""
    mark = "❌ " if call.is_error else ""
    return (
        f"<details><summary>{mark}<code>{call.tool}</code> {_gist(call.input)}</summary>\n\n"
        f"{_input(call.tool, call.input)}\n{_output(call.output)}</details>"
    )


def _gist(argument: dict[str, object]) -> str:
    """The one line that says what this call was, for a summary nobody opens."""
    for key in ("command", "pattern", "query", *PATH_KEYS):
        if value := argument.get(key):
            return f"<code>{_clip(str(value), 90)}</code>"
    return ""


def _input(name: str, argument: dict[str, object]) -> str:
    """Fenced as whatever it is, because highlighting is the point of the medium."""
    if language := FENCES.get(name):
        return _fence(language, str(argument.get("command", "")))
    if body_text := argument.get("new_string") or argument.get("content"):
        return _fence(_language(argument), str(body_text))
    return _fence("json", orjson.dumps(argument, option=orjson.OPT_INDENT_2).decode())


def _language(argument: dict[str, object]) -> str:
    for key in PATH_KEYS:
        if path := argument.get(key):
            return LANGUAGES.get(Path(str(path)).suffix, "")
    return ""


def _output(output: str) -> str:
    return _fence("", _clip(output, 4_000)) + "\n" if output else ""


def _fence(language: str, text: str) -> str:
    return f"```{language}\n{text.rstrip()}\n```\n"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n\n… truncated, {len(text) - limit:,} more characters"


def _footer(summary: dict[str, Any] | None) -> str:
    """What the turn cost, which is the one thing the transcript does not hold."""
    if not summary:
        return ""
    cost = summary.get("total_cost_usd")
    turns = summary.get("num_turns")
    bits = [b for b in (f"{turns} steps" if turns else "", f"${cost:.2f}" if cost else "") if b]
    return f"<sub>{' · '.join(bits)}</sub>" if bits else ""


def _fit(text: str) -> str:
    """Last resort. Each result is clipped already; this is for a turn of many."""
    if len(text) <= LIMIT:
        return text
    return text[: LIMIT - MARGIN] + "\n\n<sub>… comment truncated to fit GitHub's limit.</sub>"
