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
    agent-sessions cat claude:7e90a7c6 --last        # only the turn I just had, as markdown
    agent-sessions resume claude:7e90a7c6            # pick it back up
    agent-sessions resume claude:7e90a7c6@9f3c1d20   # pick it back up from a point in it
    agent-sessions resume claude:7e90a7c6 --remote    # pick it back up on my phone
    agent-sessions rename claude:7e90a7c6 doing      # a title of my own
    agent-sessions forget 7e90a7c6                   # one that was not worth keeping
    agent-sessions issue claude:7e90a7c6             # carry it on from anywhere
    agent-sessions attend                            # answer what I leave there
    agent-sessions serve                             # the same, in a browser
    agent-sessions serve --lan                       # ... and on my phone, via a QR code
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
in the terminal. That is what this index is for. `serve --lan` puts it where a phone can
reach it and prints a QR code of the address, so getting there is pointing a
camera at the terminal rather than reading an address out. Everything on that
network can then reach every conversation you have had, and resume one, which
is why it is asked for and not the default.

For the session you are in right now, none of this is needed: set
`remoteControlAtStartup` in `~/.claude/settings.json` and everything you start
is already on your phone. Remote control needs a claude.ai subscription; on an
account without one it refuses, which is why the control channel below exists.

## The control channel

A phone in a park is not on the local network, so the connection inverts.
Nothing listens here: `attend` polls a private GitHub repo, runs the turns left
there, and posts back what came of them.

    agent-sessions issue claude:7e90a7c6    # a session becomes an issue
    agent-sessions attend                   # answer prompts left on it
    agent-sessions serve --lan --attend     # both, one thing to leave running

An issue is a session — its body says which, in a table read back off the page
— and a comment of mine on it is a prompt. GitHub is the reader: markdown with
syntax highlighting, tool calls folded into `<details>`, a notification when
the answer lands, and an app to read it in. Dictate into the comment box and
you never touch a keyboard.

A turn appends to the session's own transcript, so `sync`, `search`, `show` and
`cat` see this work as they see the rest. There is no second history. A session
open in a pane at home is taken over: two agents on one transcript fork it and
then fight over which branch is live, and I am not at that keyboard.

The transcript is the truth about the conversation — it is what the agent
resumes from — and the issue is a projection of it. The body lists the turns in
the session, not the comments in the thread, so a prompt that never ran does
not appear there claiming to have happened.

👎 on either half of an exchange rewinds: the session is forked at the point
recorded in the answer, the issue is repointed at the fork, and what was
rewound over is deleted. Nothing is destroyed — the fork leaves the original
session whole, so the history is in the transcript, which is why the thread
does not need to keep it. Then post the prompt you meant.

`agent-work[bot]` is what the loop posts as, `<!-- agent-work:turn -->` marks
what it wrote before there was a bot, and 👀 marks a prompt it has taken up.
All of it lives on GitHub, not in the index, which is safe to delete — losing
it must not make the loop read its own output back as prompts.

Who posts is not cosmetic. As me, a reply was my own activity and GitHub does
not notify you about that, so the answer landing was silent and reloading the
page was the only way to see it. Set these and the loop is the app instead:

    export AGENT_WORK_APP_ID=...          # the app's id
    export AGENT_WORK_APP_KEY=~/.agent-sessions/agent-work.pem

Create a GitHub App under your own account with **Issues: read & write** and no
webhook, generate a private key, and install it on the control repo alone.
Unset, the token is whatever `gh` is signed in as, and nothing notifies.

GitHub has no long-poll, but a conditional request that has not changed costs
nothing against the rate limit, so the interval is a free choice.

### What a prompt may do

Whatever I may. A turn runs under my own settings, so a comment on the issue is
allowed exactly what a prompt typed at the keyboard is allowed — which, given
`defaultMode: bypassPermissions`, is everything. What stands between a comment
and this machine is who can reach the repo, and nothing else.

Two more things to know before leaving it running. Tool output is posted to
GitHub, and it holds what a commit never would; `redact` catches the token
shapes it can, and that is not a guarantee. And the laptop has to be awake and
online — `caffeinate -s`.

A session need not be picked up where it was left. `tree` and `show --turns`
print the point each stretch and each turn ended at, and `<id>@<point>` resumes
from there — before a compaction, or before the turn that went wrong. An agent
can only resume a session at its leaf, so this writes the session that ends at
that point, exactly as a fork is written, and resumes that. The session it came
from is not touched.

The same points name a turn to read rather than resume. `cat <id> --last` is
the exchange just had — what I asked and everything that followed — and
`cat <id>@<point>` is the one that point falls in. It is the whole `cat`,
narrowed: the same markdown, the same `--tools`, so a turn reads out of the
session exactly as it reads inside it. That is the thing worth handing to
someone; the session rarely is.

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
