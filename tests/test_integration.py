"""The one thing the unit tests cannot see: whether an agent actually came up.

Every other test stubs `wormhole.run` and asserts on what it was told. The
bugs worth catching live past that line — the wrong directory, keys sent to a
pane before its shell exists, a dead pane focused forever — and they all look
the same from here: a terminal, and no agent in it.

So this drives the real thing: a wormhole daemon of its own, on its own tmux
socket, so nothing here touches the tmux server the day's work is in. Modelled
on wormhole's `WormholeTest`, which does the same for the same reason.

STUB. The fixtures are written but have never run green; the test below says
what it must assert. To finish it:

  - decide how the daemon is built and found (`cargo build` in ~/src/wormhole,
    or a WORMHOLE_BINARY env var), and fail loudly rather than skipping if it
    is not there;
  - register the fixture repo as a project, the way `create_project` does;
  - work out how long to wait for the pane, and poll rather than sleep.

    AGENT_SESSIONS_INTEGRATION=1 uv run pytest tests/test_integration.py
"""

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("AGENT_SESSIONS_INTEGRATION"),
    reason="drives a real wormhole daemon and tmux server; set AGENT_SESSIONS_INTEGRATION=1",
)

PORT = 7119
SOCKET = f"agent-sessions-test-{PORT}"


@pytest.fixture
def tmux():
    """A tmux server of our own. The day's work is on a different socket."""

    def run(*args: str) -> str:
        out = subprocess.run(
            ["tmux", "-L", SOCKET, *args], capture_output=True, text=True, check=False
        )
        return out.stdout.strip()

    yield run
    run("kill-server")


@pytest.fixture
def fake_claude(tmp_path: Path) -> Path:
    """A stand-in for the agent, so the assertion is about the pane, not about Claude.

    It records the directory it was started in and the arguments it was given,
    then sits there, which is what a resumed session looks like from tmux.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "claude.log"
    script = bin_dir / "claude"
    script.write_text(f'#!/bin/sh\necho "$PWD $*" >> {log}\nexec sleep 300\n')
    script.chmod(0o755)
    return log


@pytest.fixture
def daemon(tmp_path: Path, tmux):
    """Wormhole, in that tmux server, with its own port and worktree directory."""
    raise NotImplementedError("see the module docstring")


def test_resume_starts_the_agent_in_the_directory_the_session_was_had_in(
    daemon, fake_claude: Path
) -> None:
    """The whole class of bug in one assertion: an agent is running, on that session.

    Sync a transcript filed under a directory that is not the project's working
    tree, hit `/resume/<id>`, and wait for the agent to come up. What has to be
    true: `fake_claude` was started in the transcript's own directory, with
    `-r <native id>`, and nothing in the pane says the session was not found.
    """
    raise NotImplementedError("see the module docstring")
