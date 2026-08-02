# senderos

Index and resume the work you have done with coding agents.

A *sendero* is one unit of that work: a durable, branching, resumable piece of it.
Not a "conversation" — the dialogue is the interface, not the substance.

Design: https://github.com/dandavison/log/issues/289

## Use

    senderos sync                              # bring the index up to date
    senderos search "worktree relocation"      # find senderos by what was said
    senderos ls -p wormhole --since 2w         # what was I doing on this worktree
    senderos show claude:7e90a7c6 --turns      # my turns, with context size at each
    senderos tree claude:7e90a7c6              # branch, compaction and fork topology
    senderos cat claude:7e90a7c6 --tools       # the whole thing, tool calls included
    senderos resume claude:7e90a7c6            # pick it back up
    senderos skills add                        # teach an agent the command surface

Output adapts to who is asking: an aligned table for a terminal, TSV with
nothing truncated for a coding agent, `--json` for anything else. Hints and
errors go to stderr, so only data reaches stdout.

## Develop

    uv run pytest
    uv run ruff format . && uv run ruff check .
    uv run ty check

Requires [wormhole](https://github.com/dandavison/wormhole) to be running: it owns
project identity and terminal actuation. If it is down, senderos fails rather than
degrading.
