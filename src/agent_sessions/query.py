"""Reading the index. Nothing here writes."""

import sqlite3
import time
from dataclasses import dataclass

# Weeks are as long as anyone needs; months and years vary and would need a calendar.
UNITS = {"h": 3600, "d": 86400, "w": 604800}

SUFFIXES = {"k": 1_000, "m": 1_000_000}

COLUMNS = (
    "id, project, agent, title, model, started_at, ended_at,"
    " n_user_turns, n_messages, context_tokens, output_tokens, dropped_tokens, cwd, git_branch"
)

# What each sort orders by, and which way round it reads best: newest, largest
# and longest first, but a name A to Z.
SORTS = {
    "recent": "ended_at",
    "project": "project",
    "context": "context_tokens",
    "turns": "n_user_turns",
    "title": "title",
}

ASCENDING = frozenset({"project", "title"})


def order_by(sort: str, reverse: bool = False) -> str:
    """Sessions missing the value sort last either way: they are not the answer."""
    column = SORTS[sort]
    descending = (sort not in ASCENDING) != reverse
    return f"{column} IS NULL, {column} {'DESC' if descending else 'ASC'}"


@dataclass(frozen=True, slots=True)
class Filters:
    project: str | None = None
    agent: str | None = None
    since: str | None = None
    min_context: str | None = None

    def where(self) -> tuple[str, list[object]]:
        clauses, values = [], []
        if self.project:
            # `-p temporal` should find its tasks too, since a closed one attributes
            # to the bare repo while a live one keeps its branch.
            clauses.append("(project = ? OR project LIKE ?)")
            values += [self.project, f"{self.project}:%"]
        if self.agent:
            clauses.append("agent = ?")
            values.append(self.agent)
        if self.since:
            clauses.append("ended_at >= ?")
            values.append(int(time.time()) - parse_duration(self.since))
        if self.min_context:
            clauses.append("context_tokens >= ?")
            values.append(parse_count(self.min_context))
        return (" AND ".join(clauses) or "1", values)


def parse_duration(text: str) -> int:
    """`2w`, `36h`, `10d`."""
    if len(text) < 2 or text[-1] not in UNITS or not text[:-1].isdigit():
        raise ValueError(f"bad duration {text!r}: expected a number then one of h, d, w (e.g. 2w)")
    return int(text[:-1]) * UNITS[text[-1]]


def parse_count(text: str) -> int:
    """`500k`, `1m`, `20000`."""
    scale = SUFFIXES.get(text[-1:].lower(), 1)
    digits = text[:-1] if scale > 1 else text
    if not digits.isdigit():
        raise ValueError(f"bad count {text!r}: expected a number, optionally suffixed k or m")
    return int(digits) * scale


def recent(
    conn: sqlite3.Connection,
    filters: Filters,
    sort: str,
    limit: int,
    reverse: bool = False,
    offset: int = 0,
) -> list[dict]:
    where, values = filters.where()
    rows = conn.execute(
        f"SELECT {COLUMNS} FROM session WHERE {where}"
        f" ORDER BY {order_by(sort, reverse)} LIMIT ? OFFSET ?",
        [*values, limit, offset],
    )
    return [dict(row) for row in rows]


def search(
    conn: sqlite3.Connection, query: str, filters: Filters, limit: int, offset: int = 0
) -> list[dict]:
    """Sessions with a node matching `query`, best match first.

    Ranking blends BM25 with recency: what I am looking for is nearly always
    something recent, and a five-month-old exact match is rarely the one. A
    session is scored by its single best node, and shows that node as the snippet.
    """
    where, values = filters.where()
    rows = conn.execute(
        f"""
        WITH matched AS (
            -- bm25() may only be called where the fts table is queried directly,
            -- so ranking and picking the best node per session are separate steps.
            SELECT node.session_id AS sid, node.text AS text, bm25(node_fts) AS rank
            FROM node_fts
            JOIN node ON node.rowid = node_fts.rowid
            WHERE node_fts MATCH ?
        ),
        best AS (
            SELECT sid, text, rank,
                   row_number() OVER (PARTITION BY sid ORDER BY rank) AS n
            FROM matched
        )
        SELECT {COLUMNS}, best.text AS snippet, best.rank AS rank
        FROM best
        JOIN session ON session.id = best.sid
        WHERE best.n = 1 AND {where}
        ORDER BY best.rank * (1 + {_RECENCY}) ASC
        LIMIT ? OFFSET ?
        """,
        [query, *values, limit, offset],
    )
    return [dict(row) for row in rows]


# bm25 is negative and lower is better, so the boost multiplies: a recent hit
# becomes more negative and sorts first. The boost decays over months, halving
# at about 30 days old.
_RECENCY = "3.0 / (1.0 + (strftime('%s', 'now') - COALESCE(session.ended_at, 0)) / 2592000.0)"


def worked_in(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """The projects worked in and the tasks of them, as wormhole names both.

    A project is the part before the colon and a task is the whole key. Filtering
    by a project takes in its tasks; filtering by a task is one of them. The
    filter offers both, so both are counted here.
    """
    keys = {
        row["project"]
        for row in conn.execute("SELECT DISTINCT project FROM session WHERE project != ''")
        if row["project"]
    }
    return sorted({k.split(":")[0] for k in keys}), sorted(k for k in keys if ":" in k)


def get(conn: sqlite3.Connection, id: str) -> dict | None:
    """Resolve a session by full id, bare native id, or any unambiguous prefix."""
    rows = conn.execute(
        f"SELECT {COLUMNS}, native_id, leaf_uuid, path FROM session"
        " WHERE id = ? OR native_id = ? OR id LIKE ? OR native_id LIKE ?"
        " LIMIT 2",
        (id, id, f"{id}%", f"{id}%"),
    ).fetchall()
    if len(rows) != 1:
        return None
    return dict(rows[0])


def split_point(id: str) -> tuple[str, str]:
    """`<session>@<point>`: which session, and where in it to pick up."""
    session, _, at = id.partition("@")
    return session, at


def node(conn: sqlite3.Connection, session_id: str, at: str) -> dict | None:
    """Resolve a point in a session by uuid or any unambiguous prefix."""
    rows = conn.execute(
        "SELECT uuid, role, text, ts FROM node WHERE session_id = ? AND uuid LIKE ? LIMIT 2",
        (session_id, f"{at}%"),
    ).fetchall()
    if len(rows) != 1:
        return None
    return dict(rows[0])


def session_of(conn: sqlite3.Connection, at: str) -> str | None:
    """The session a point belongs to, for when one is handed over on its own.

    A point is only addressable inside a session, so an id that turns out to be
    one is a mistake with an answer rather than a session that is not there.
    """
    rows = conn.execute(
        "SELECT DISTINCT session_id FROM node WHERE uuid LIKE ? LIMIT 2", (f"{at}%",)
    ).fetchall()
    return str(rows[0]["session_id"]) if len(rows) == 1 else None


def turns(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """What I said, in order, with how large the context had grown by then.

    A user record carries no usage of its own — only the model's replies do —
    so each turn borrows the context of the next reply, which is the first one
    to have counted what I just said.
    """
    rows = conn.execute(
        """
        SELECT uuid, role, ts, text, is_branch_point,
               (SELECT reply.context_tokens FROM node reply
                 WHERE reply.session_id = node.session_id
                   AND reply.seq >= node.seq
                   AND reply.context_tokens > 0
                 ORDER BY reply.seq LIMIT 1) AS context_tokens
        FROM node
        WHERE session_id = ? AND text != '' AND role IN ('user', 'summary')
        ORDER BY seq
        """,
        (session_id,),
    )
    return [dict(row) for row in rows]
