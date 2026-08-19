# agent-sessions

Find and resume the sessions you have had with coding agents.

Design: https://github.com/dandavison/log/issues/289

## Use

    agent-sessions sync                              # bring the index up to date
    agent-sessions search "worktree relocation"      # find agent-sessions by what was said
    agent-sessions ls -p wormhole --since 2w         # what was I doing on this worktree
    agent-sessions show claude:7e90a7c6 --turns      # my turns, with context size at each
    agent-sessions tree claude:7e90a7c6              # branch, compaction and fork topology
    agent-sessions cat claude:7e90a7c6 --tools       # the whole thing, tool calls included
    agent-sessions resume claude:7e90a7c6            # pick it back up
    agent-sessions resume claude:7e90a7c6@9f3c1d20   # pick it back up from a point in it
    agent-sessions resume claude:7e90a7c6 --remote    # pick it back up on my phone
    agent-sessions rename claude:7e90a7c6 doing      # a title of my own
    agent-sessions forget 7e90a7c6                   # one that was not worth keeping
    agent-sessions serve                             # the same, in a browser
    agent-sessions skills add                        # teach an agent the command surface

Output adapts to who is asking: an aligned table for a terminal, TSV with
nothing truncated for a coding agent, `--json` for anything else. Hints and
errors go to stderr, so only data reaches stdout.

`serve` puts the index in a browser, where resuming is a GET:
`http://localhost:7118/resume/claude:7e90a7c6` picks the session back up. So a
link is enough — from the page, a note, a chat message, or an agent's output.

Add `?remote=1` and the session is started with the agent's own remote control
on, and the browser is sent after it. This is for the phone. Claude Code can
already put the session in front of you on a phone; what it cannot do is find
one from three weeks ago and pick that up, because its session picker runs only
in the terminal. That is what this index is for. Serve it on an address the
phone can reach — `agent-sessions serve --host 100.x.x.x` on a tailnet — and
finding old work and carrying it on are both a tap.

For the session you are in right now, none of this is needed: set
`remoteControlAtStartup` in `~/.claude/settings.json` and everything you start
is already on your phone.

A session need not be picked up where it was left. `tree` and `show --turns`
print the point each stretch and each turn ended at, and `<id>@<point>` resumes
from there — before a compaction, or before the turn that went wrong. An agent
can only resume a session at its leaf, so this writes the session that ends at
that point, exactly as a fork is written, and resumes that. The session it came
from is not touched.

Not every session was worth having. `forget` takes one out of the index and its
transcript out of the agent's reach — to `~/.agent-sessions/forgotten`, so being
wrong about that costs a move back — along with the prompts it left at the
agent's input line. A running session is refused, because its agent still has
the file open; the id to hand it is the one Claude prints as it exits, and that
works before any sync has seen the session.

`rename` writes the title where the agent keeps its own, so it survives a sync
and shows in the agent's UI too.

## Develop

    uv run pytest
    AGENT_SESSIONS_INTEGRATION=1 uv run pytest tests/test_integration.py  # own tmux server
    uv run ruff format . && uv run ruff check .
    uv run ty check
    ./rename-project agent-sessions <new-name>   # if it needs another name

Requires [wormhole](https://github.com/dandavison/wormhole) to be running: it owns
project identity and terminal actuation. If it is down, agent-sessions fails rather than
degrading.
