# agent-sessions

Indexes Claude Code sessions. Search, inspect, resume — from terminal, browser,
or GitHub issue.


## Install

    uv sync
    uv run agent-sessions sync

## Commands

    agent-sessions sync                                 # update the index
    agent-sessions search "worktree relocation"         # search session content
    agent-sessions ls -p wormhole --since 2w             # list sessions, filtered
    agent-sessions show claude:7e90a7c6 --turns           # turns, with context size
    agent-sessions tree claude:7e90a7c6                    # branch/compaction/fork topology
    agent-sessions cat claude:7e90a7c6 --tools              # full transcript incl. tool calls
    agent-sessions cat claude:7e90a7c6 --last                # most recent turn, as markdown
    agent-sessions resume claude:7e90a7c6                     # resume
    agent-sessions resume claude:7e90a7c6@9f3c1d20             # resume from an earlier point
    agent-sessions resume claude:7e90a7c6 --remote               # resume via remote control (phone)
    agent-sessions rename claude:7e90a7c6 doing                   # set a title
    agent-sessions forget 7e90a7c6                                  # remove from index
    agent-sessions issue claude:7e90a7c6 / attend                    # GitHub control channel, below
    agent-sessions serve [--lan]                                      # web UI
    agent-sessions agent [--with pi|qwen] [prompt]                      # start an agent with this skill loaded
    agent-sessions skills add                                            # install the skill

Output: table (terminal), TSV (coding agent), or `--json`.

## Web UI

`agent-sessions serve` — browser view of the index, resume by click.
`http://localhost:7118/resume/claude:7e90a7c6` resumes directly. `--lan`:
serve on LAN, print a QR code for phone access. `?remote=1` on a resume link:
hand off to Claude Code remote control.


## Develop

    uv run pytest
    AGENT_SESSIONS_INTEGRATION=1 uv run pytest tests/test_integration.py  # own tmux server
    uv run ruff format . && uv run ruff check .
    uv run ty check
    ./rename-project agent-sessions <new-name>   # if it needs another name

Requires [wormhole](https://github.com/dandavison/wormhole) running: it owns
project identity and terminal actuation. If down, agent-sessions fails rather
than degrading.
