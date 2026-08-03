"""The shape of a session: where it branched, where it was compacted, what forked off it.

A transcript is a DAG, and most of it is uninteresting: long single-child runs
where nothing happened but the work. Those collapse into one segment each, so
what is left to look at is the branching.
"""

import sqlite3
from dataclasses import dataclass, field


@dataclass(slots=True)
class Segment:
    """A run of nodes with nothing to choose between them."""

    start: str
    end: str
    turns: int
    messages: int
    started_at: int | None
    ended_at: int | None
    context_tokens: int
    leads_to_leaf: bool = False
    abandoned: bool = False
    compaction: dict | None = None
    children: list["Segment"] = field(default_factory=list)


@dataclass(slots=True)
class Topology:
    roots: list[Segment]
    forks: list[dict]
    forked_from: dict | None


def of(conn: sqlite3.Connection, session: dict) -> Topology:
    nodes = [
        dict(row)
        for row in conn.execute(
            "SELECT uuid, parent_uuid, seq, role, text, ts, context_tokens, is_branch_point"
            " FROM node WHERE session_id = ? ORDER BY seq",
            (session["id"],),
        )
    ]
    compactions = {
        row["uuid"]: dict(row)
        for row in conn.execute("SELECT * FROM compaction WHERE session_id = ?", (session["id"],))
    }
    roots = _segments(nodes, compactions)
    _mark_active(roots, _ancestry(nodes, session.get("leaf_uuid")))
    return Topology(
        roots=roots,
        forks=[
            dict(r)
            for r in conn.execute(
                "SELECT child, at_uuid FROM session_edge WHERE parent = ? AND kind = 'fork'",
                (session["id"],),
            )
        ],
        forked_from=next(
            (
                dict(r)
                for r in conn.execute(
                    "SELECT parent, at_uuid FROM session_edge WHERE child = ? AND kind = 'fork'",
                    (session["id"],),
                )
            ),
            None,
        ),
    )


def _segments(nodes: list[dict], compactions: dict[str, dict]) -> list[Segment]:
    by_uuid = {n["uuid"]: n for n in nodes}
    children: dict[str, list[dict]] = {}
    for node in nodes:
        parent = node["parent_uuid"]
        if parent in by_uuid:
            children.setdefault(parent, []).append(node)

    roots = [n for n in nodes if n["parent_uuid"] not in by_uuid]
    return [_walk(root, children, compactions) for root in roots]


def _walk(start: dict, children: dict[str, list[dict]], compactions: dict[str, dict]) -> Segment:
    """Follow single children until the thread ends or divides."""
    run = [start]
    while len(kids := children.get(run[-1]["uuid"], [])) == 1:
        run.append(kids[0])

    segment = _summarise(run, compactions)
    segment.children = [
        _walk(child, children, compactions) for child in children.get(run[-1]["uuid"], [])
    ]
    return segment


def _summarise(run: list[dict], compactions: dict[str, dict]) -> Segment:
    stamps = [n["ts"] for n in run if n["ts"]]
    contexts = [n["context_tokens"] for n in run if n["context_tokens"]]
    return Segment(
        start=run[0]["uuid"],
        end=run[-1]["uuid"],
        turns=sum(1 for n in run if n["role"] == "user" and n["text"]),
        messages=len(run),
        started_at=min(stamps, default=None),
        ended_at=max(stamps, default=None),
        context_tokens=contexts[-1] if contexts else 0,
        compaction=compactions.get(run[0]["uuid"]),
    )


def _ancestry(nodes: list[dict], leaf_uuid: str | None) -> set[str]:
    """Every node on the live thread, walking parents back from the leaf."""
    by_uuid = {n["uuid"]: n for n in nodes}
    node = by_uuid.get(leaf_uuid or "") or (nodes[-1] if nodes else None)
    seen: set[str] = set()
    while node and node["uuid"] not in seen:
        seen.add(node["uuid"])
        node = by_uuid.get(node["parent_uuid"] or "")
    return seen


def _mark_active(segments: list[Segment], live: set[str]) -> None:
    """Work out which branches lost, and which were merely left behind.

    A segment was abandoned only if a sibling leads to the leaf and it does
    not: that is a rewind, and somebody chose the other way. Roots are not
    siblings in that sense — compaction starts a new one, and everything under
    the previous root was compacted away rather than abandoned.
    """
    for segment in segments:
        _mark_reachable(segment, live)
    _mark_losers(segments, siblings=False)


def _mark_reachable(segment: Segment, live: set[str]) -> bool:
    reached = [_mark_reachable(child, live) for child in segment.children]
    segment.leads_to_leaf = segment.end in live or any(reached)
    return segment.leads_to_leaf


def _mark_losers(segments: list[Segment], siblings: bool) -> None:
    if siblings and len(segments) > 1:
        # The branch that leads to the leaf was taken. In an era compaction has
        # since closed off none of them do, and then the one written last was
        # taken, because rewinding appends.
        taken = next((s for s in segments if s.leads_to_leaf), segments[-1])
        for segment in segments:
            segment.abandoned = segment is not taken
    for segment in segments:
        _mark_losers(segment.children, siblings=True)
