"""Client for wormhole, which owns project identity and terminal actuation.

Wormhole is a hard dependency: if it is not running we fail, we do not degrade.
"""

import os
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import httpx

PORT = int(os.environ.get("WORMHOLE_PORT", "7117"))
BASE = f"http://localhost:{PORT}"


class WormholeUnavailable(Exception):
    def __init__(self) -> None:
        super().__init__(f"wormhole is not running at {BASE}. Run `wormhole server start` first.")


@dataclass(frozen=True, slots=True)
class Worktree:
    project_key: str
    working_tree: str
    repo: str = ""
    branch: str | None = None


def worktrees() -> list[Worktree]:
    return [
        Worktree(
            project_key=w["project_key"],
            working_tree=w["working_tree"],
            repo=w["repo"],
            branch=w["branch"],
        )
        for w in _get("/project/worktrees")
    ]


def run(project: str, cwd: str, command: str, tag: str, pid: int | None = None) -> None:
    """Wormhole owns the terminal, so it is what actually runs the command.

    It knows about projects and panes and nothing about agents: the command line
    is ours to compose, and so is `cwd` — the directory the session was had in,
    which is where the agent has to be started to find it, and not necessarily
    the project's working tree. `tag` names what the pane is for, so asking twice
    lands in the same pane; given the pid of something already running, wormhole
    focuses its pane instead.
    """
    params = {"project": project, "cwd": cwd, "cmd": command, "tag": tag}
    if pid:
        params["pid"] = str(pid)
    _post("/terminal/run", params)


class Attributor:
    """Maps an arbitrary path to the project whose working tree contains it.

    Longest match wins, so a task's worktree beats the repo it was cut from.

    A quarter of the corpus was worked on in worktrees that have since been
    removed, and wormhole only knows about live ones. Those paths still name
    their repo — worktrees are laid out `<worktree_dir>/<repo>/<branch>/<repo>`
    — so a closed task is attributed to the bare repo. The worktree directory
    is read off wormhole's own live worktrees rather than assumed, since it is
    configurable.
    """

    def __init__(self, worktrees: list[Worktree]) -> None:
        self._prefixes = sorted({(w.working_tree.rstrip("/"), w.project_key) for w in worktrees})
        self._keys = [p for p, _ in self._prefixes]
        self._repos = {w.repo for w in worktrees if w.repo}
        self._worktree_dir = _worktree_dir(worktrees)

    def project_for(self, path: str | None) -> str | None:
        if not path:
            return None
        path = path.rstrip("/")
        i = bisect_right(self._keys, path)
        for prefix, project in reversed(self._prefixes[:i]):
            if path == prefix or path.startswith(prefix + "/"):
                return project
        return self._repo_of_closed_task(path)

    def _repo_of_closed_task(self, path: str) -> str | None:
        if not self._worktree_dir:
            return None
        try:
            parts = Path(path).relative_to(self._worktree_dir).parts
        except ValueError:
            return None
        return parts[0] if parts and parts[0] in self._repos else None


def _worktree_dir(worktrees: list[Worktree]) -> Path | None:
    """`<worktree_dir>/<repo>/<branch>/<repo>`, so three levels up from any task."""
    for w in worktrees:
        if w.branch:
            return Path(w.working_tree).parents[2]
    return None


def _get(route: str) -> list[dict[str, str]]:
    try:
        response = httpx.get(BASE + route, timeout=10.0)
    except httpx.ConnectError as e:
        raise WormholeUnavailable() from e
    response.raise_for_status()
    return response.json()


def _post(route: str, params: dict[str, str]) -> None:
    try:
        response = httpx.post(BASE + route, params=params, timeout=30.0)
    except httpx.ConnectError as e:
        raise WormholeUnavailable() from e
    response.raise_for_status()
