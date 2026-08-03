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
    agent-sessions skills add                        # teach an agent the command surface

Output adapts to who is asking: an aligned table for a terminal, TSV with
nothing truncated for a coding agent, `--json` for anything else. Hints and
errors go to stderr, so only data reaches stdout.

## Develop

    uv run pytest
    uv run ruff format . && uv run ruff check .
    uv run ty check

Requires [wormhole](https://github.com/dandavison/wormhole) to be running: it owns
project identity and terminal actuation. If it is down, agent-sessions fails rather than
degrading.
