"""The CLI is the product, and its job is to cost an agent as few calls as possible."""

from pathlib import Path

import pytest

from agent_sessions import cli, db, query
from agent_sessions.models import Node, Session

ID = "claude:7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "index.db"
    conn = db.connect(path)
    db.write_session(
        conn,
        Session(
            id=ID,
            agent="claude",
            native_id=ID.split(":")[1],
            path="/t/a.jsonl",
            project="wormhole",
            title="why is conform relocating worktrees",
            ended_at=1_785_000_000,
            n_user_turns=1,
        ),
    )
    db.write_nodes(
        conn,
        [
            Node(ID, "u1", None, 0, "user", 1, "why is conform relocating worktrees"),
            Node(
                ID,
                "a1",
                "u1",
                1,
                "assistant",
                2,
                "Because it reconciles submodules.",
                context_tokens=1000,
            ),
        ],
    )
    db.reindex_fts(conn)
    conn.commit()
    conn.close()
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


@pytest.fixture
def run(capsys: pytest.CaptureFixture[str]):
    """Drive the real entry point, so exit codes are part of what is tested."""

    def invoke(*args: str) -> tuple[int, str, str]:
        capsys.readouterr()
        code = cli.run(list(args))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return invoke


# --- nothing tells the agent to sync ---------------------------------------


@pytest.mark.parametrize(
    "args",
    [("search", "relocating"), ("ls",), ("show", ID), ("tree", ID), ("status",)],
)
def test_no_command_suggests_syncing(indexed: Path, run, args: tuple[str, ...]) -> None:
    """Syncing is the user's call. Suggesting it costs the agent a tool call."""
    _, out, err = run(*args)
    assert "agent-sessions sync" not in out + err


def test_status_still_reports_staleness_as_a_fact(indexed: Path, run) -> None:
    _, out, _ = run("status")
    assert "changed_since" in out


# --- every result names the next command, ready to run ---------------------


@pytest.mark.parametrize("args", [("search", "relocating"), ("ls",)])
def test_finding_something_names_the_next_command(
    indexed: Path, run, args: tuple[str, ...]
) -> None:
    _, _, err = run(*args)
    assert f"Next: agent-sessions show {ID} --turns" in err


def test_show_names_what_comes_after_it(indexed: Path, run) -> None:
    _, _, err = run("show", ID)
    assert f"agent-sessions cat {ID} --tools" in err


def test_the_next_command_carries_a_real_id_not_a_placeholder(indexed: Path, run) -> None:
    """An agent can run a worked example; it cannot run a placeholder."""
    _, _, err = run("search", "relocating")
    assert "<id>" not in err
    assert ID in err


def test_the_next_command_is_one_this_tool_accepts(indexed: Path, run) -> None:
    """Follow the hint literally and it must work."""
    _, _, err = run("search", "relocating")
    suggested = err.split("Next: agent-sessions ")[1].split("   (")[0].split()
    code, out, _ = run(*suggested)
    assert code == 0
    assert "relocating" in out


def test_other_verbs_are_named_without_repeating_the_id(indexed: Path, run) -> None:
    _, _, err = run("search", "relocating")
    assert "(also: cat --tools, tree, resume)" in err
    assert err.count(ID) == 1


# --- hints stay off stdout -------------------------------------------------


def test_hints_never_pollute_the_data(indexed: Path, run) -> None:
    _, out, err = run("search", "relocating")
    assert "Next:" in err
    assert "Next:" not in out


def test_quiet_emits_ids_alone(indexed: Path, run) -> None:
    _, out, _ = run("search", "relocating", "-q")
    assert out.strip() == ID


def test_nothing_found_exits_one(indexed: Path, run) -> None:
    code, _, err = run("search", "kangaroo")
    assert code == cli.EXIT_NO_RESULTS
    assert "Next:" not in err


# --- read commands do no filesystem sweep ----------------------------------


def test_reads_do_not_stat_the_corpus(indexed: Path, run, monkeypatch: pytest.MonkeyPatch) -> None:
    """Staleness cost 5x the query it decorated, for advice we no longer give."""

    def fail(*args: object, **kwargs: object) -> int:
        raise AssertionError("a read command swept the filesystem")

    monkeypatch.setattr(cli.index, "stale", fail)
    assert run("search", "relocating")[0] == 0
    assert run("ls")[0] == 0
    assert run("show", ID)[0] == 0


def test_status_is_where_staleness_belongs(
    indexed: Path, run, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.index, "stale", lambda conn: 7)
    _, out, _ = run("status")
    assert "7" in out


# --- ids round-trip --------------------------------------------------------


def test_a_prefix_is_accepted_everywhere(indexed: Path, run) -> None:
    for command in ("show", "tree"):
        assert run(command, "7e90a7c6")[0] == 0


def test_turns_carry_the_point_to_resume_at(indexed: Path, run) -> None:
    """A turn is only addressable if its uuid is printed next to it."""
    _, out, _ = run("show", ID, "--turns")
    assert "u1" in out


def test_the_shape_names_the_point_each_stretch_ends_at(indexed: Path, run) -> None:
    _, out, _ = run("tree", ID)
    assert "a1" in out


def test_resuming_at_a_point_that_does_not_exist_is_a_usage_error(indexed: Path, run) -> None:
    code, _, err = run("resume", f"{ID}@nope")
    assert code == cli.EXIT_USAGE
    assert "nope" in err


def test_query_filters_reach_the_index(indexed: Path, run) -> None:
    assert run("search", "relocating", "-p", "wormhole")[0] == 0
    assert run("search", "relocating", "-p", "temporal")[0] == cli.EXIT_NO_RESULTS


def test_a_malformed_duration_is_a_usage_error(indexed: Path, run) -> None:
    code, _, err = run("ls", "--since", "2y")
    assert code == cli.EXIT_USAGE
    assert "2y" in err


def test_sorts_are_offered_by_name(indexed: Path, run) -> None:
    for sort in query.SORTS:
        assert run("ls", "--sort", sort)[0] == 0


def test_status_counts_sessions_not_the_tool(indexed: Path, run) -> None:
    """The row is a count of sessions; naming it after the command reads as nonsense."""
    _, out, _ = run("status")
    assert "sessions " in out
    assert "agent-sessions " not in out
