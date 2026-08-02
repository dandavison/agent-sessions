import httpx
import pytest

from senderos.wormhole import Attributor, Worktree, WormholeUnavailable, worktrees


def attributor(*pairs: tuple[str, str]) -> Attributor:
    return Attributor([Worktree(project_key=k, working_tree=p) for p, k in pairs])


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
