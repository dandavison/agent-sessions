"""Client for wormhole, which owns project identity and terminal actuation.

Wormhole is a hard dependency: if it is not running we fail, we do not degrade.
"""

import os
from bisect import bisect_right
from dataclasses import dataclass

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


def worktrees() -> list[Worktree]:
    return [
        Worktree(project_key=w["project_key"], working_tree=w["working_tree"])
        for w in _get("/project/worktrees")
    ]


def resume(sendero_id: str, fork: bool = False) -> None:
    _post(f"/conversations/resume/{sendero_id}", params={"fork": "true"} if fork else {})


class Attributor:
    """Maps an arbitrary path to the project whose working tree contains it.

    Longest match wins, so a task's worktree beats the repo it was cut from.
    Sorting once and binary-searching keeps this cheap across ~100k lookups.
    """

    def __init__(self, worktrees: list[Worktree]) -> None:
        self._prefixes = sorted({(w.working_tree.rstrip("/"), w.project_key) for w in worktrees})
        self._keys = [p for p, _ in self._prefixes]

    def project_for(self, path: str | None) -> str | None:
        if not path:
            return None
        path = path.rstrip("/")
        i = bisect_right(self._keys, path)
        for prefix, project in reversed(self._prefixes[:i]):
            if path == prefix or path.startswith(prefix + "/"):
                return project
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
