"""The control channel: a private GitHub repo where an issue is a session.

Nothing listens on this machine. Everything here is an outbound request, which
is the point — a socket that can start an agent is exactly what I did not want
open while I am in the park.

GitHub has no long-poll, so a comment is noticed by asking. A conditional
request that has not changed returns 304 and costs nothing against the rate
limit, so the interval is a free choice rather than a budget: the ETag is not
an optimisation to add later, it is what makes the design work at all.

The comments are posted with my own token, so the author never distinguishes
mine from its own. A marker in the body does, and the eyes on a comment record
that it has been taken up. Both live on GitHub deliberately: the index is a
derived cache that is safe to delete, and losing it must not make the loop read
its own output back as prompts and never stop.
"""

import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_sessions import comment, log

API = "https://api.github.com"
REPO = os.environ.get("AGENT_WORK_REPO", "dandavison/agent-work")

# Invisible once GitHub renders it, and the only thing that says a comment is
# not a prompt.
MARKER = "<!-- agent-work:turn -->"

# Posted the moment a prompt is picked up, so that a turn taking minutes still
# shows something within seconds of asking for it.
SEEN = "eyes"


class NoToken(Exception):
    def __init__(self) -> None:
        super().__init__("No GitHub token. Run `gh auth login`.")


@dataclass(frozen=True, slots=True)
class Comment:
    id: int
    body: str
    author: str
    taken_up: bool = False

    @property
    def is_ours(self) -> bool:
        return MARKER in self.body


@dataclass(frozen=True, slots=True)
class Issue:
    number: int
    title: str
    body: str
    url: str = ""

    @property
    def session_id(self) -> str:
        return comment.session_id(self.body)


@dataclass
class Channel:
    """One private repo, and the conversation going on in its issues."""

    repo: str = REPO
    client: httpx.Client | None = None
    _etags: dict[str, str] = field(default_factory=dict, init=False)
    _cached: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.Client(base_url=API, headers=_headers(), timeout=30.0)

    # --- reading ------------------------------------------------------------

    def issues(self) -> list[Issue]:
        """Open issues, which are the sessions being worked on.

        GitHub returns pull requests from this endpoint too, and a pull request
        is not a session.
        """
        found = self._poll(f"/repos/{self.repo}/issues", {"state": "open", "per_page": "100"})
        return [_issue(raw) for raw in found if "pull_request" not in raw]

    def comments(self, number: int) -> list[Comment]:
        found = self._poll(f"/repos/{self.repo}/issues/{number}/comments", {"per_page": "100"})
        return [_comment(raw) for raw in found]

    # --- writing ------------------------------------------------------------

    def post(self, number: int, body: str) -> None:
        """Everything posted carries the marker. Nothing else keeps the loop finite."""
        self._send(
            "POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": f"{MARKER}\n{body}"}
        )

    def take_up(self, comment_id: int) -> None:
        self._send(
            "POST", f"/repos/{self.repo}/issues/comments/{comment_id}/reactions", {"content": SEEN}
        )

    def set_body(self, number: int, body: str) -> None:
        self._send("PATCH", f"/repos/{self.repo}/issues/{number}", {"body": body})

    def open(self, title: str, body: str) -> Issue:
        raw = self._send("POST", f"/repos/{self.repo}/issues", {"title": title, "body": body})
        return _issue(raw)

    # --- the wire -----------------------------------------------------------

    def _poll(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        """A conditional GET. A 304 means unchanged, which is not the same as empty."""
        assert self.client is not None
        headers = {"If-None-Match": tag} if (tag := self._etags.get(path)) else {}
        response = self.client.get(path, params=params, headers=headers)
        log.detail(f"GET {path} -> {response.status_code}")
        if response.status_code == 304:
            return self._cached.get(path, [])
        response.raise_for_status()
        if tag := response.headers.get("ETag"):
            self._etags[path] = tag
        self._cached[path] = response.json()
        return self._cached[path]

    def _send(self, method: str, path: str, payload: dict[str, str]) -> dict[str, Any]:
        assert self.client is not None
        response = self.client.request(method, path, json=payload)
        log.detail(f"{method} {path} -> {response.status_code}")
        response.raise_for_status()
        return response.json() if response.content else {}


def _issue(raw: dict[str, Any]) -> Issue:
    return Issue(
        number=raw["number"],
        title=raw.get("title", ""),
        body=raw.get("body") or "",
        url=raw.get("html_url", ""),
    )


def _comment(raw: dict[str, Any]) -> Comment:
    return Comment(
        id=raw["id"],
        body=raw.get("body") or "",
        author=(raw.get("user") or {}).get("login", ""),
        taken_up=bool((raw.get("reactions") or {}).get(SEEN)),
    )


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _token() -> str:
    """Whatever `gh` is already signed in as. One place to be authenticated."""
    try:
        found = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise NoToken() from e
    return found.stdout.strip()
