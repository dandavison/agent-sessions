"""Output. Humans and agents want opposite things from the same command.

A human wants an aligned table truncated to the terminal, colour, and a hint about
what to do next. An agent wants every value in full, no ANSI, nothing truncated,
and structure it can split on. No command formats its own output; they all come
through here.
"""

import json
import os
import shutil
import sys
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

# Set by the agents that drive this tool. Presence of any of them means the caller
# is not a human at a terminal.
AGENT_ENV_VARS = (
    "CLAUDECODE",
    "CLAUDE_CODE",
    "CODEX_SANDBOX",
    "CURSOR_AGENT",
    "GEMINI_CLI",
    "AI_AGENT",
)


class Format(StrEnum):
    HUMAN = "human"
    AGENT = "agent"
    JSON = "json"
    QUIET = "quiet"


def detect() -> Format:
    if any(os.environ.get(v) for v in AGENT_ENV_VARS):
        return Format.AGENT
    return Format.HUMAN if sys.stdout.isatty() else Format.AGENT


class Renderer:
    def __init__(self, fmt: Format) -> None:
        self.fmt = fmt

    def table(self, rows: Sequence[dict[str, Any]], quiet_key: str = "id") -> None:
        """Emit a list of records. The shape differs per format; the data does not."""
        if self.fmt is Format.JSON:
            self._json(rows)
        elif self.fmt is Format.QUIET:
            for row in rows:
                print(row[quiet_key])
        elif self.fmt is Format.AGENT:
            self._tsv(rows)
        else:
            self._aligned(rows)

    def record(self, obj: dict[str, Any]) -> None:
        """Emit a single record."""
        if self.fmt is Format.JSON:
            self._json(obj)
        elif self.fmt is Format.QUIET:
            print(obj.get("id", ""))
        else:
            width = max((len(k) for k in obj), default=0)
            for key, value in obj.items():
                print(f"{key:<{width}}  {_cell(value, truncate=False)}")

    def line(self, text: str) -> None:
        """Emit one line of already-formatted data."""
        if self.fmt is not Format.QUIET:
            print(text)

    def hint(self, text: str) -> None:
        """Name the next command. Never on stdout: the agent is parsing that."""
        if self.fmt in (Format.HUMAN, Format.AGENT):
            print(f"Hint: {text}", file=sys.stderr)

    def warn(self, text: str) -> None:
        print(f"Warning: {text}", file=sys.stderr)

    def _json(self, payload: object) -> None:
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")

    def _tsv(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        columns = list(rows[0])
        print("\t".join(columns))
        for row in rows:
            print("\t".join(_cell(row.get(c), truncate=False) for c in columns))

    def _aligned(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            print("(none)")
            return
        columns = list(rows[0])
        budget = _column_budget(len(columns))
        cells = [[_cell(row.get(c), truncate=True, limit=budget) for c in columns] for row in rows]
        widths = [
            max(len(c.upper()), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)
        ]
        print("  ".join(c.upper().ljust(w) for c, w in zip(columns, widths, strict=True)).rstrip())
        for row in cells:
            print("  ".join(v.ljust(w) for v, w in zip(row, widths, strict=True)).rstrip())


def _column_budget(n_columns: int) -> int:
    """Per-column width that keeps a table inside the terminal."""
    width = shutil.get_terminal_size((100, 24)).columns
    return max(8, (width - 2 * n_columns) // n_columns)


def _cell(value: Any, truncate: bool, limit: int = 0) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\t", " ")
    if truncate and limit and len(text) > limit:
        return text[: limit - 1] + "…"
    return text
