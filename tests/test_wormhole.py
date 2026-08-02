import httpx
import pytest

from senderos.wormhole import Attributor, Worktree, WormholeUnavailable, worktrees


def attributor(*pairs: tuple[str, str]) -> Attributor:
    return Attributor([Worktree(project_key=k, working_tree=p) for p, k in pairs])


def with_worktrees() -> Attributor:
    """One live task and the repo it was cut from, laid out as wormhole lays them out."""
    return Attributor(
        [
            Worktree("temporal", "/Users/dan/src/temporalio/temporal", "temporal", None),
            Worktree("delta", "/Users/dan/src/delta", "delta", None),
            Worktree(
                "temporal:dan/live",
                "/Users/dan/worktrees/temporal/dan--live/temporal",
                "temporal",
                "dan/live",
            ),
        ]
    )


def test_longest_prefix_wins() -> None:
    a = attributor(
        ("/Users/dan/src/temporalio/temporal", "temporal"),
        ("/Users/dan/worktrees/temporal/dan--x/temporal", "temporal:dan/x"),
    )
    assert a.project_for("/Users/dan/worktrees/temporal/dan--x/temporal/host") == "temporal:dan/x"
    assert a.project_for("/Users/dan/src/temporalio/temporal/host") == "temporal"


def test_respects_path_boundaries() -> None:
    a = attributor(("/Users/dan/src/foo", "foo"))
    assert a.project_for("/Users/dan/src/foobar") is None
    assert a.project_for("/Users/dan/src/foo") == "foo"
    assert a.project_for("/Users/dan/src/foo/bar") == "foo"


def test_unknown_path_is_unattributed() -> None:
    a = attributor(("/Users/dan/src/foo", "foo"))
    assert a.project_for("/tmp/scratch") is None
    assert a.project_for(None) is None


def test_trailing_slashes_do_not_matter() -> None:
    a = attributor(("/Users/dan/src/foo/", "foo"))
    assert a.project_for("/Users/dan/src/foo") == "foo"
    assert a.project_for("/Users/dan/src/foo/bar/") == "foo"


def test_nested_worktrees_pick_the_innermost() -> None:
    a = attributor(
        ("/a", "outer"),
        ("/a/b", "middle"),
        ("/a/b/c", "inner"),
    )
    assert a.project_for("/a/b/c/d") == "inner"
    assert a.project_for("/a/b/x") == "middle"
    assert a.project_for("/a/x") == "outer"


def test_wormhole_down_is_an_error_not_a_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> object:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", refuse)
    with pytest.raises(WormholeUnavailable):
        worktrees()


def test_closed_task_attributes_to_its_repo() -> None:
    """The worktree is gone, so wormhole cannot name the task — but the path names the repo."""
    a = with_worktrees()
    assert a.project_for("/Users/dan/worktrees/temporal/dan--closed/temporal/host") == "temporal"


def test_live_task_still_beats_its_repo() -> None:
    a = with_worktrees()
    assert a.project_for("/Users/dan/worktrees/temporal/dan--live/temporal") == "temporal:dan/live"


def test_worktree_of_an_unknown_repo_is_unattributed() -> None:
    a = with_worktrees()
    assert a.project_for("/Users/dan/worktrees/notmine/some--branch/notmine") is None


def test_paths_outside_the_worktree_dir_are_unaffected() -> None:
    a = with_worktrees()
    assert a.project_for("/Users/dan/.cargo/registry/src/index.crates.io/parking_lot-0.9") is None
    assert a.project_for("/tmp/scratch") is None


def test_repo_attribution_needs_a_live_task_to_locate_the_worktree_dir() -> None:
    """With no task anywhere, the worktree directory is unknown and nothing is guessed."""
    a = Attributor([Worktree("delta", "/Users/dan/src/delta", "delta", None)])
    assert a.project_for("/Users/dan/worktrees/delta/gone/delta") is None
