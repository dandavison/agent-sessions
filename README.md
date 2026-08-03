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
    agent-sessions serve                             # the same, in a browser
    agent-sessions skills add                        # teach an agent the command surface

Output adapts to who is asking: an aligned table for a terminal, TSV with
nothing truncated for a coding agent, `--json` for anything else. Hints and
errors go to stderr, so only data reaches stdout.

`serve` puts the index in a browser, where resuming is a GET:
`http://localhost:7118/resume/claude:7e90a7c6` picks the session back up, and
`?fork=1` branches it instead. So a link is enough — from the page, a note, a
chat message, or an agent's output.

## Develop

    uv run pytest
    uv run ruff format . && uv run ruff check .
    uv run ty check
    ./rename-project agent-sessions <new-name>   # if it needs another name

Requires [wormhole](https://github.com/dandavison/wormhole) to be running: it owns
project identity and terminal actuation. If it is down, agent-sessions fails rather than
degrading.
