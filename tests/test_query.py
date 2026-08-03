import time
from pathlib import Path

import pytest

from agent_sessions import db, query
from agent_sessions.models import Node, Session

NOW = int(time.time())
DAY = 86400


def session(id: str, **kw) -> Session:
    return Session(
        id=id,
        agent=kw.pop("agent", "claude"),
        native_id=id.split(":")[1],
        path=f"/t/{id}.jsonl",
        ended_at=kw.pop("ended_at", NOW),
        **kw,
    )


def populate(conn, sessions: list[Session], nodes: list[Node]) -> None:
    for s in sessions:
        db.write_session(conn, s)
    db.write_nodes(conn, nodes)
    db.reindex_fts(conn)
    conn.commit()


@pytest.fixture
def conn(tmp_path: Path):
    return db.connect(tmp_path / "index.db")


# --- duration and count parsing --------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"), [("36h", 129_600), ("10d", 864_000), ("2w", 1_209_600)]
)
def test_parse_duration(text: str, seconds: int) -> None:
    assert query.parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["2", "w", "2y", "two_w", "", "-1d"])
def test_parse_duration_rejects_nonsense(text: str) -> None:
    with pytest.raises(ValueError):
        query.parse_duration(text)


@pytest.mark.parametrize(
    ("text", "count"), [("500k", 500_000), ("1m", 1_000_000), ("20000", 20_000), ("1M", 1_000_000)]
)
def test_parse_count(text: str, count: int) -> None:
    assert query.parse_count(text) == count


@pytest.mark.parametrize("text", ["", "k", "lots", "1.5m"])
def test_parse_count_rejects_nonsense(text: str) -> None:
    with pytest.raises(ValueError):
        query.parse_count(text)


# --- filters ---------------------------------------------------------------


def three(conn) -> None:
    populate(
        conn,
        [
            session("claude:a", project="wormhole", context_tokens=500_000),
            session("claude:b", project="temporal:dan/x", ended_at=NOW - 30 * DAY),
            session("claude:c", project="temporal", context_tokens=10_000),
        ],
        [],
    )


def ids(rows: list[dict]) -> set[str]:
    return {r["id"] for r in rows}


def test_project_filter_includes_tasks_of_that_repo(conn) -> None:
    three(conn)
    found = query.recent(conn, query.Filters(project="temporal"), "recent", 10)
    assert ids(found) == {"claude:b", "claude:c"}


def test_project_filter_is_not_a_prefix_match(conn) -> None:
    """`-p temporal` must not sweep in a differently named repo like temporalio."""
    populate(conn, [session("claude:d", project="temporalio")], [])
    assert query.recent(conn, query.Filters(project="temporal"), "recent", 10) == []


def test_since_filter(conn) -> None:
    three(conn)
    found = query.recent(conn, query.Filters(since="2w"), "recent", 10)
    assert ids(found) == {"claude:a", "claude:c"}


def test_min_context_filter(conn) -> None:
    three(conn)
    found = query.recent(conn, query.Filters(min_context="100k"), "recent", 10)
    assert ids(found) == {"claude:a"}


def test_sorts(conn) -> None:
    three(conn)
    by_context = query.recent(conn, query.Filters(), "context", 10)
    assert [r["id"] for r in by_context][0] == "claude:a"
    by_recent = query.recent(conn, query.Filters(), "recent", 10)
    assert [r["id"] for r in by_recent][-1] == "claude:b"


def test_limit(conn) -> None:
    three(conn)
    assert len(query.recent(conn, query.Filters(), "recent", 2)) == 2


# --- search ----------------------------------------------------------------


def corpus(conn) -> None:
    populate(
        conn,
        [
            session("claude:old", project="wormhole", ended_at=NOW - 150 * DAY),
            session("claude:new", project="wormhole"),
            session("claude:other", project="temporal"),
        ],
        [
            Node("claude:old", "o1", None, 0, "user", NOW, "worktree relocation, at length"),
            Node("claude:old", "o2", "o1", 1, "assistant", NOW, "relocation relocation relocation"),
            Node("claude:new", "n1", None, 0, "user", NOW, "worktree relocation once"),
            Node("claude:other", "t1", None, 0, "user", NOW, "something else entirely"),
            Node("claude:other", "t2", "t1", 1, "assistant", NOW, ""),
        ],
    )


def test_search_finds_matching_sessions(conn) -> None:
    corpus(conn)
    assert ids(query.search(conn, "relocation", query.Filters(), 10)) == {
        "claude:old",
        "claude:new",
    }


def test_search_returns_one_row_per_session(conn) -> None:
    """`old` matches twice; it should still appear once, with its best node."""
    corpus(conn)
    hits = query.search(conn, "relocation", query.Filters(), 10)
    assert len(hits) == len({h["id"] for h in hits})


def test_recency_outranks_a_denser_old_match(conn) -> None:
    """A five-month-old match is rarely the one being looked for."""
    corpus(conn)
    hits = query.search(conn, "relocation", query.Filters(), 10)
    assert hits[0]["id"] == "claude:new"


def test_search_respects_filters(conn) -> None:
    corpus(conn)
    assert query.search(conn, "relocation", query.Filters(project="temporal"), 10) == []


def test_search_carries_the_matching_text_as_a_snippet(conn) -> None:
    corpus(conn)
    hits = query.search(conn, "relocation", query.Filters(project="wormhole"), 10)
    assert all("relocation" in h["snippet"] for h in hits)


def test_search_supports_phrases(conn) -> None:
    corpus(conn)
    assert ids(query.search(conn, '"worktree relocation"', query.Filters(), 10)) == {
        "claude:old",
        "claude:new",
    }


def test_search_matches_nothing_when_nothing_matches(conn) -> None:
    corpus(conn)
    assert query.search(conn, "kangaroo", query.Filters(), 10) == []


# --- resolving an id -------------------------------------------------------


def found_id(conn, given: str) -> str:
    session = query.get(conn, given)
    assert session is not None
    return session["id"]


def test_get_by_full_id(conn) -> None:
    three(conn)
    assert found_id(conn, "claude:a") == "claude:a"


def test_get_by_bare_native_id(conn) -> None:
    three(conn)
    assert found_id(conn, "a") == "claude:a"


def test_get_by_unambiguous_prefix(conn) -> None:
    populate(conn, [session("claude:7daebc42-3ff5")], [])
    assert found_id(conn, "claude:7dae") == "claude:7daebc42-3ff5"
    assert found_id(conn, "7dae") == "claude:7daebc42-3ff5"


def test_get_refuses_an_ambiguous_prefix(conn) -> None:
    populate(conn, [session("claude:ab1"), session("claude:ab2")], [])
    assert query.get(conn, "claude:ab") is None


def test_get_of_something_absent(conn) -> None:
    three(conn)
    assert query.get(conn, "claude:nope") is None


# --- turns -----------------------------------------------------------------


def test_turns_are_mine_in_order(conn) -> None:
    populate(
        conn,
        [session("claude:a")],
        [
            Node("claude:a", "u1", None, 0, "user", 1, "first"),
            Node("claude:a", "a1", "u1", 1, "assistant", 2, "a reply"),
            Node("claude:a", "u2", "a1", 2, "user", 3, "second"),
            Node("claude:a", "t1", "u2", 3, "assistant", 4, ""),
        ],
    )
    assert [t["text"] for t in query.turns(conn, "claude:a")] == ["first", "second"]


def test_a_compact_summary_counts_as_a_turn(conn) -> None:
    """It is the only surviving record of everything compaction dropped."""
    populate(
        conn,
        [session("claude:a")],
        [
            Node("claude:a", "s1", None, 0, "summary", 1, "Summary: earlier work on worktrees."),
            Node("claude:a", "u1", "s1", 1, "user", 2, "carry on"),
        ],
    )
    assert [t["role"] for t in query.turns(conn, "claude:a")] == ["summary", "user"]


def test_each_turn_reports_the_context_of_the_reply_it_drew(conn) -> None:
    """A user record has no usage of its own; only the model's replies do."""
    populate(
        conn,
        [session("claude:a")],
        [
            Node("claude:a", "u1", None, 0, "user", 1, "first"),
            Node("claude:a", "a1", "u1", 1, "assistant", 2, "reply", context_tokens=1000),
            Node("claude:a", "u2", "a1", 2, "user", 3, "second"),
            Node("claude:a", "a2", "u2", 3, "assistant", 4, "reply", context_tokens=5000),
        ],
    )
    assert [t["context_tokens"] for t in query.turns(conn, "claude:a")] == [1000, 5000]


def test_a_turn_with_no_reply_yet_has_no_context(conn) -> None:
    populate(
        conn,
        [session("claude:a")],
        [Node("claude:a", "u1", None, 0, "user", 1, "asked, never answered")],
    )
    assert query.turns(conn, "claude:a")[0]["context_tokens"] is None


def test_an_old_match_still_wins_if_recent_ones_are_much_worse(conn) -> None:
    """Recency is a thumb on the scale, not an override."""
    populate(
        conn,
        [
            session("claude:old", ended_at=NOW - 150 * DAY),
            session("claude:new"),
        ],
        [
            Node("claude:old", "o1", None, 0, "user", NOW, "kangaroo kangaroo kangaroo kangaroo"),
            Node(
                "claude:new",
                "n1",
                None,
                0,
                "user",
                NOW,
                "kangaroo " + " ".join(f"filler{i}" for i in range(400)),
            ),
        ],
    )
    assert query.search(conn, "kangaroo", query.Filters(), 10)[0]["id"] == "claude:old"


# --- how matching behaves --------------------------------------------------


def stemming_corpus(conn) -> None:
    populate(
        conn,
        [session("claude:a")],
        [Node("claude:a", "u1", None, 0, "user", NOW, "Relocating the Worktrees")],
    )


@pytest.mark.parametrize("q", ["relocating", "Relocating", "RELOCATING", "ReLoCaTiNg"])
def test_search_is_case_insensitive(conn, q: str) -> None:
    stemming_corpus(conn)
    assert len(query.search(conn, q, query.Filters(), 10)) == 1


@pytest.mark.parametrize("q", ["relocate", "relocated", "relocation", "relocating"])
def test_search_is_stemmed(conn, q: str) -> None:
    stemming_corpus(conn)
    assert len(query.search(conn, q, query.Filters(), 10)) == 1


def test_a_prefix_must_target_the_stem(conn) -> None:
    """ "Relocating" is stored as `reloc`, so a longer prefix cannot match it."""
    stemming_corpus(conn)
    assert len(query.search(conn, "reloc*", query.Filters(), 10)) == 1
    assert query.search(conn, "relocat*", query.Filters(), 10) == []


def test_bare_words_are_anded(conn) -> None:
    stemming_corpus(conn)
    assert len(query.search(conn, "relocating worktrees", query.Filters(), 10)) == 1
    assert query.search(conn, "relocating kangaroo", query.Filters(), 10) == []


def test_or_and_not(conn) -> None:
    stemming_corpus(conn)
    assert len(query.search(conn, "kangaroo OR worktrees", query.Filters(), 10)) == 1
    assert query.search(conn, "worktrees NOT relocating", query.Filters(), 10) == []
