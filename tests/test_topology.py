from pathlib import Path

import pytest

from agent_sessions import db, topology
from agent_sessions.models import Compaction, Edge, Node, Session

SID = "claude:a"


@pytest.fixture
def conn(tmp_path: Path):
    return db.connect(tmp_path / "index.db")


def build(conn, nodes: list[Node], leaf: str | None = None, **kw) -> dict:
    db.write_session(
        conn,
        Session(id=SID, agent="claude", native_id="a", path="/t/a.jsonl", leaf_uuid=leaf, **kw),
    )
    db.write_nodes(conn, nodes)
    conn.commit()
    return {"id": SID, "leaf_uuid": leaf}


def node(uuid: str, parent: str | None, seq: int, role: str = "user", text: str = "x") -> Node:
    return Node(SID, uuid, parent, seq, role, seq, text)


def test_a_straight_thread_is_one_segment(conn) -> None:
    session = build(conn, [node("u1", None, 0), node("a1", "u1", 1), node("u2", "a1", 2)], "u2")
    (root,) = topology.of(conn, session).roots
    assert (root.start, root.end, root.children) == ("u1", "u2", [])


def test_a_segment_counts_only_turns_that_were_typed(conn) -> None:
    session = build(
        conn,
        [
            node("u1", None, 0),
            node("a1", "u1", 1, role="assistant", text="reply"),
            node("t1", "a1", 2, role="assistant", text=""),
        ],
        "t1",
    )
    (root,) = topology.of(conn, session).roots
    assert (root.turns, root.messages) == (1, 3)


def test_a_branch_divides_the_tree(conn) -> None:
    session = build(
        conn,
        [node("u1", None, 0), node("x", "u1", 1), node("y", "u1", 2)],
        "y",
    )
    (root,) = topology.of(conn, session).roots
    assert root.end == "u1"
    assert [c.start for c in root.children] == ["x", "y"]


def test_the_branch_leading_to_the_leaf_was_taken(conn) -> None:
    session = build(conn, [node("u1", None, 0), node("x", "u1", 1), node("y", "u1", 2)], "x")
    (root,) = topology.of(conn, session).roots
    taken, dropped = root.children[0], root.children[1]
    assert (taken.abandoned, dropped.abandoned) == (False, True)


def test_with_no_leaf_among_them_the_last_written_was_taken(conn) -> None:
    """Rewinding appends, so in an era compaction has closed off, latest wins."""
    session = build(
        conn,
        [node("r1", None, 0), node("x", "r1", 1), node("y", "r1", 2), node("r2", None, 3)],
        "r2",
    )
    first_era = topology.of(conn, session).roots[0]
    assert [c.abandoned for c in first_era.children] == [True, False]


def test_compaction_starts_a_root_of_its_own(conn) -> None:
    """The boundary record has no parent, so there is no path back across it."""
    session = build(conn, [node("u1", None, 0), node("c1", None, 1), node("u2", "c1", 2)], "u2")
    db.write_compactions(
        conn,
        [Compaction(SID, "c1", 1, "auto", 900_000, 20_000, "u1", "s1", 7)],
    )
    conn.commit()

    roots = topology.of(conn, session).roots
    assert len(roots) == 2
    assert roots[0].compaction is None
    assert roots[1].compaction is not None
    assert roots[1].compaction["trigger"] == "auto"


def test_an_earlier_era_is_not_abandoned(conn) -> None:
    """It was compacted away, which is not the same as being rewound past."""
    session = build(conn, [node("u1", None, 0), node("c1", None, 1), node("u2", "c1", 2)], "u2")
    roots = topology.of(conn, session).roots
    assert [r.abandoned for r in roots] == [False, False]


def test_context_of_a_segment_is_its_last_known_value(conn) -> None:
    nodes = [node("u1", None, 0), node("a1", "u1", 1), node("a2", "a1", 2)]
    nodes[1].context_tokens = 1000
    nodes[2].context_tokens = 5000
    session = build(conn, nodes, "a2")
    assert topology.of(conn, session).roots[0].context_tokens == 5000


def test_forks_out_are_listed(conn) -> None:
    session = build(conn, [node("u1", None, 0)], "u1")
    db.write_edges(conn, [Edge(child="claude:b", parent=SID, kind="fork", at_uuid="u1")])
    conn.commit()
    assert topology.of(conn, session).forks == [{"child": "claude:b", "at_uuid": "u1"}]


def test_the_fork_it_came_from_is_named(conn) -> None:
    session = build(conn, [node("u1", None, 0)], "u1")
    db.write_edges(conn, [Edge(child=SID, parent="claude:parent", kind="fork", at_uuid="u1")])
    conn.commit()
    assert topology.of(conn, session).forked_from == {"parent": "claude:parent", "at_uuid": "u1"}


def test_a_session_with_no_forks(conn) -> None:
    session = build(conn, [node("u1", None, 0)], "u1")
    shape = topology.of(conn, session)
    assert (shape.forks, shape.forked_from) == ([], None)


def test_a_node_whose_parent_is_missing_starts_a_root(conn) -> None:
    """Relocation and other surgery can leave a chain hanging."""
    session = build(conn, [node("u1", None, 0), node("orphan", "gone", 1)], "orphan")
    assert [r.start for r in topology.of(conn, session).roots] == ["u1", "orphan"]
