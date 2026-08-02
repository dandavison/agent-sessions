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

SORTS = {"recent": "ended_at DESC", "context": "context_tokens DESC", "turns": "n_user_turns DESC"}


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


def recent(conn: sqlite3.Connection, filters: Filters, sort: str, limit: int) -> list[dict]:
    where, values = filters.where()
    rows = conn.execute(
        f"SELECT {COLUMNS} FROM sendero WHERE {where} ORDER BY {SORTS[sort]} LIMIT ?",
        [*values, limit],
    )
    return [dict(row) for row in rows]


def search(conn: sqlite3.Connection, query: str, filters: Filters, limit: int) -> list[dict]:
    """Senderos with a node matching `query`, best match first.

    Ranking blends BM25 with recency: what I am looking for is nearly always
    something recent, and a five-month-old exact match is rarely the one. A
    sendero is scored by its single best node, and shows that node as the snippet.
    """
    where, values = filters.where()
    rows = conn.execute(
        f"""
        WITH matched AS (
            -- bm25() may only be called where the fts table is queried directly,
            -- so ranking and picking the best node per sendero are separate steps.
            SELECT node.sendero_id AS sid, node.text AS text, bm25(node_fts) AS rank
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
        JOIN sendero ON sendero.id = best.sid
        WHERE best.n = 1 AND {where}
        ORDER BY best.rank * (1 + {_RECENCY}) ASC
        LIMIT ?
        """,
        [query, *values, limit],
    )
    return [dict(row) for row in rows]


# bm25 is negative and lower is better, so the boost multiplies: a recent hit
# becomes more negative and sorts first. The boost decays over months, halving
# at about 30 days old.
_RECENCY = "3.0 / (1.0 + (strftime('%s', 'now') - COALESCE(sendero.ended_at, 0)) / 2592000.0)"


def get(conn: sqlite3.Connection, id: str) -> dict | None:
    """Resolve a sendero by full id, bare native id, or any unambiguous prefix."""
    rows = conn.execute(
        f"SELECT {COLUMNS}, leaf_uuid, path FROM sendero"
        " WHERE id = ? OR native_id = ? OR id LIKE ? OR native_id LIKE ?"
        " LIMIT 2",
        (id, id, f"{id}%", f"{id}%"),
    ).fetchall()
    if len(rows) != 1:
        return None
    return dict(rows[0])


def turns(conn: sqlite3.Connection, sendero_id: str) -> list[dict]:
    """What I said, in order, with how large the context had grown by then.

    A user record carries no usage of its own — only the model's replies do —
    so each turn borrows the context of the next reply, which is the first one
    to have counted what I just said.
    """
    rows = conn.execute(
        """
        SELECT uuid, role, ts, text, is_branch_point,
               (SELECT reply.context_tokens FROM node reply
                 WHERE reply.sendero_id = node.sendero_id
                   AND reply.seq >= node.seq
                   AND reply.context_tokens > 0
                 ORDER BY reply.seq LIMIT 1) AS context_tokens
        FROM node
        WHERE sendero_id = ? AND text != '' AND role IN ('user', 'summary')
        ORDER BY seq
        """,
        (sendero_id,),
    )
    return [dict(row) for row in rows]
