"""How a session is spelled for a person, shared by the terminal and the browser.

Where `render` decides the shape of a record set — aligned, TSV, JSON — this
decides how one value reads: a token count, a date, the one line that describes
a stretch of a transcript.
"""

import time

from agent_sessions import topology


def date(when: int | None, with_time: bool = False) -> str:
    if not when:
        return ""
    shape = "%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d"
    return time.strftime(shape, time.localtime(when))


def tokens(n: int | None) -> str:
    if not n:
        return ""
    return f"{n / 1_000_000:.1f}m" if n >= 1_000_000 else f"{n // 1000}k" if n >= 1000 else str(n)


def plural(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def row(s: dict) -> dict:
    """One session, as a line in a list of them."""
    return {
        "id": s["id"],
        "project": s["project"],
        "when": date(s["ended_at"]),
        "turns": s["n_user_turns"],
        "context": tokens(s["context_tokens"]),
        "title": s["title"],
    }


def details(s: dict) -> dict:
    """One session, in full."""
    return {
        "id": s["id"],
        "title": s["title"],
        "project": s["project"],
        "branch": s["git_branch"],
        "cwd": s["cwd"],
        "model": s["model"],
        "started": date(s["started_at"]),
        "ended": date(s["ended_at"]),
        "turns": s["n_user_turns"],
        "messages": s["n_messages"],
        "context": tokens(s["context_tokens"]),
        "output": tokens(s["output_tokens"]),
        "dropped": tokens(s["dropped_tokens"]),
    }


def turn(t: dict) -> dict:
    return {
        "at": point(t["uuid"]),
        "when": date(t["ts"], with_time=True),
        "role": t["role"],
        "context": tokens(t["context_tokens"]),
        "text": t["text"],
    }


def snippet(text: str, query_text: str) -> str:
    """The matching node, trimmed to the neighbourhood of the first matching word."""
    flat = " ".join(text.split())
    words = [w.strip('"*').lower() for w in query_text.split() if w.isalnum()]
    lowered = flat.lower()
    at = next((i for w in words if (i := lowered.find(w)) >= 0), 0)
    start = max(0, at - 40)
    return ("…" if start else "") + flat[start : start + 200]


def point(uuid: str | None) -> str:
    """Where in a session something happened, short enough to type after an id."""
    return (uuid or "")[:8]


def describe(segment: topology.Segment) -> str:
    """One stretch of a transcript: what happened in it, or what compaction did."""
    if c := segment.compaction:
        dropped = (c["pre_tokens"] or 0) - (c["post_tokens"] or 0)
        return (
            f"compacted ({c['trigger']}) {tokens(c['pre_tokens'])} → {tokens(c['post_tokens'])},"
            f" dropped {tokens(dropped)}, {c['preserved_count']} kept"
        )
    parts = [
        date(segment.started_at, with_time=True),
        plural(segment.turns, "turn") if segment.turns else plural(segment.messages, "message"),
        tokens(segment.context_tokens),
        "(abandoned)" if segment.abandoned else "",
    ]
    return "  ".join(p for p in parts if p)
