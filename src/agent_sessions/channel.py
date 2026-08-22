"""The control channel: a private GitHub repo where an issue is a session.

Nothing listens on this machine. Everything here is an outbound request, which
is the point — a socket that can start an agent is exactly what I did not want
open while I am in the park.

GitHub has no long-poll, so a comment is noticed by asking. A conditional
request that has not changed returns 304 and costs nothing against the rate
limit, so the interval is a free choice rather than a budget: the ETag is not
an optimisation to add later, it is what makes the design work at all.

Who posts matters more than it looks. With my own token the author of a reply
is me, and GitHub does not notify you about your own activity — so the answer
landing was silent, and reloading the page was the only way to find out. A
GitHub App posts as `agent-work[bot]`, which is someone else, so the ordinary
notification fires. Configure `AGENT_WORK_APP_ID` and `AGENT_WORK_APP_KEY` and
it is used; otherwise the token is whatever `gh` is signed in as.

Telling my comments from its own then has two answers, and needs both: the
author settles it for anything the app posted, and the marker still settles it
for everything posted back when it was me. The eyes on a comment record that it
has been taken up. All of it lives on GitHub deliberately: the index is a
derived cache that is safe to delete, and losing it must not make the loop read
its own output back as prompts and never stop.
"""

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import jwt

from agent_sessions import comment, log

# Comment-body syntax, so they are defined where comments are written.
from agent_sessions.comment import MARKER, RUNNING

API = "https://api.github.com"
REPO = os.environ.get("AGENT_WORK_REPO", "dandavison/agent-work")

# Posted the moment a prompt is picked up, so that a turn taking minutes still
# shows something within seconds of asking for it.
SEEN = "eyes"

# Swapped in when the turn lands. The eyes were added on pickup and never
# removed, so they said "seen at some point" and nothing about whether the
# thing was still working. Now their presence means in flight.
DONE = "rocket"

# One tap, on my prompt or on the answer to it, meaning: put the session back
# the way it was before this. Cheap to notice — the reaction summary comes back
# with every comment already.
REWIND = "-1"

# The app, when there is one. Named rather than discovered, so that running
# without it is a choice and not an accident.
APP_ID = "AGENT_WORK_APP_ID"
APP_KEY = "AGENT_WORK_APP_KEY"

# GitHub gives an installation token an hour; minted per call it would be two
# extra requests every two seconds.
_minted: tuple[str, float] = ("", 0.0)


_ACCEPT = "application/vnd.github+json"


class NotAuthorized(Exception):
    """The credential is wrong, which is not something waiting will fix.

    Distinguished from every other failure because the loop retries those, and
    retrying this one produces an afternoon of identical lines and no work.
    """


def _refuses(response: httpx.Response) -> bool:
    """Whether GitHub is refusing us rather than merely having a bad moment.

    A 403 is also how a rate limit arrives, and that one does come right on its
    own; the remaining-count is what tells them apart.
    """
    if response.status_code == 401:
        return True
    limited = response.headers.get("x-ratelimit-remaining") == "0"
    return response.status_code == 403 and not limited


class NoToken(Exception):
    def __init__(self, why: str = "No GitHub token. Run `gh auth login`.") -> None:
        super().__init__(why)

    @classmethod
    def no_key(cls, path: str) -> "NoToken":
        return cls(f"Cannot read the app key at {path}. ${APP_KEY} names it.")

    @classmethod
    def not_installed(cls) -> "NoToken":
        return cls(f"The app in ${APP_ID} is not installed on any account.")


@dataclass(frozen=True, slots=True)
class Comment:
    id: int
    body: str
    author: str
    taken_up: bool = False
    rewind_wanted: bool = False

    @property
    def is_ours(self) -> bool:
        """Whether the loop wrote this, by either of the two things that say so.

        The author, for anything the app posted. The marker, for the comments
        posted back when this ran as me — without it, a repo full of those
        becomes a queue of prompts the moment the app is turned on.

        The marker has to be where the loop writes it, at the start. Matched
        anywhere, an answer that merely wrote *about* the markers counted as
        one, and this conversation is full of those.
        """
        return self.author.endswith("[bot]") or self._starts(MARKER) or self.is_progress

    @property
    def is_progress(self) -> bool:
        """A turn still writing into this, or one that died while it was."""
        return self._starts(RUNNING)

    def _starts(self, marker: str) -> bool:
        return self.body.lstrip().startswith(marker)

    @property
    def point(self) -> str:
        """Where the session was before the turn this comment answers."""
        return comment.point(self.body)


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
    _mine: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = httpx.Client(base_url=API, timeout=30.0)
            self._mine = True

    def _authorize(self) -> None:
        """Put the current token on the client, before every request that needs one.

        An installation token lasts an hour and this process does not. Set once
        at construction it was still the startup token sixty minutes later, and
        the loop then 401ed for ever. `token()` caches, so asking each time
        costs a comparison and refreshes itself when the cache goes stale.

        A client I was handed brings its own authorization; the tests rely on
        that, and so would anything else supplying one.
        """
        if self._mine:
            assert self.client is not None
            self.client.headers.update(_headers())

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

    def post(self, number: int, body: str) -> int:
        """Post a comment and say which one it is, so it can be written into."""
        made = self._send("POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": body})
        return int(made["id"])

    def edit(self, comment_id: int, body: str) -> None:
        """Rewrite a comment. Edits do not notify, which is what makes progress bearable."""
        self._send("PATCH", f"/repos/{self.repo}/issues/comments/{comment_id}", {"body": body})

    def delete(self, comment_id: int) -> None:
        """Rewound work is in the transcript; the thread is only the live conversation."""
        self._send("DELETE", f"/repos/{self.repo}/issues/comments/{comment_id}", {})

    def take_up(self, comment_id: int) -> None:
        self._send(
            "POST", f"/repos/{self.repo}/issues/comments/{comment_id}/reactions", {"content": SEEN}
        )

    def mark_done(self, comment_id: int) -> None:
        """Swap the eyes for a rocket: this one is answered, not merely noticed."""
        self._send(
            "POST", f"/repos/{self.repo}/issues/comments/{comment_id}/reactions", {"content": DONE}
        )
        for reaction in self._get(f"/repos/{self.repo}/issues/comments/{comment_id}/reactions"):
            if reaction.get("content") == SEEN:
                self._send(
                    "DELETE",
                    f"/repos/{self.repo}/issues/comments/{comment_id}/reactions/{reaction['id']}",
                    {},
                )

    def _get(self, path: str) -> list[dict[str, Any]]:
        assert self.client is not None
        self._authorize()
        response = self.client.get(path)
        if _refuses(response):
            raise NotAuthorized(f"GitHub refused us: {response.status_code} on {path}")
        response.raise_for_status()
        found = response.json()
        return found if isinstance(found, list) else []

    def set_body(self, number: int, body: str) -> None:
        self._send("PATCH", f"/repos/{self.repo}/issues/{number}", {"body": body})

    def open(self, title: str, body: str) -> Issue:
        raw = self._send("POST", f"/repos/{self.repo}/issues", {"title": title, "body": body})
        return _issue(raw)

    # --- the wire -----------------------------------------------------------

    def _poll(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        """A conditional GET. A 304 means unchanged, which is not the same as empty."""
        assert self.client is not None
        self._authorize()
        headers = {"If-None-Match": tag} if (tag := self._etags.get(path)) else {}
        response = self.client.get(path, params=params, headers=headers)
        log.detail(f"GET {path} -> {response.status_code}")
        if _refuses(response):
            raise NotAuthorized(f"GitHub refused us: {response.status_code} on {path}")
        if response.status_code == 304:
            return self._cached.get(path, [])
        response.raise_for_status()
        if tag := response.headers.get("ETag"):
            self._etags[path] = tag
        self._cached[path] = response.json()
        return self._cached[path]

    def _send(self, method: str, path: str, payload: dict[str, str]) -> dict[str, Any]:
        assert self.client is not None
        self._authorize()
        response = self.client.request(method, path, json=payload)
        log.detail(f"{method} {path} -> {response.status_code}")
        if _refuses(response):
            raise NotAuthorized(f"GitHub refused us: {response.status_code} on {path}")
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
        rewind_wanted=bool((raw.get("reactions") or {}).get(REWIND)),
    )


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token()}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def token() -> str:
    """The app's token if one is configured, and otherwise mine.

    Configured by name rather than discovered, so that running as myself — and
    getting no notifications — is a choice rather than something that happened.
    """
    app_id, key_path = os.environ.get(APP_ID), os.environ.get(APP_KEY)
    return _installation_token(app_id, key_path) if app_id and key_path else _gh_token()


def _installation_token(app_id: str, key_path: str) -> str:
    global _minted
    held, expires = _minted
    if held and time.time() < expires:
        return held
    key = _read_key(key_path)
    client = _new_client()
    signed = {"Authorization": f"Bearer {_jwt(app_id, key)}", "Accept": _ACCEPT}
    installations = client.get("/app/installations", headers=signed)
    installations.raise_for_status()
    found = installations.json()
    if not found:
        raise NoToken.not_installed()
    minted = client.post(f"/app/installations/{found[0]['id']}/access_tokens", headers=signed)
    minted.raise_for_status()
    # Well inside the hour GitHub gives it, so a turn never runs out mid-flight.
    _minted = (minted.json()["token"], time.time() + 45 * 60)
    log.detail(f"minted an installation token for app {app_id}")
    return _minted[0]


def _jwt(app_id: str, key: str) -> str:
    """Signed with the app's own key, and short-lived: GitHub allows ten minutes."""
    now = int(time.time())
    return jwt.encode({"iat": now - 60, "exp": now + 9 * 60, "iss": app_id}, key, algorithm="RS256")


def _read_key(key_path: str) -> str:
    try:
        return Path(key_path).expanduser().read_text()
    except OSError as e:
        raise NoToken.no_key(key_path) from e


def _new_client() -> httpx.Client:
    return httpx.Client(base_url=API, timeout=30.0)


def _gh_token() -> str:
    """Whatever `gh` is already signed in as. One place to be authenticated."""
    try:
        found = subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise NoToken() from e
    return found.stdout.strip()
