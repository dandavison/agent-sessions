"""What must never happen, in one place, checked before anything is spent.

Three different things go wrong with money here, and a single bound loose
enough to cover all three would be too loose to catch any of them. One turn can
run away inside itself. The loop can discover a pile of old prompts and work
through them, which is what happened: nine already-answered prompts queued for
replay at two dollars each. And a day of individually reasonable turns can add
up to something I would not have agreed to in advance.

So: a per-turn cap the agent enforces on itself, a rule that old prompts are
never run at all, a rate that says more than one a minute is not me, and a
daily ceiling behind all of it.

The breaker is a file. Tripping it stops the loop and nothing restarts it but
me — automatic recovery from "that spent more than it should have" is the same
bug a second time.
"""

import os
import time
from pathlib import Path

import orjson

from agent_sessions import log

HOME = Path.home() / ".agent-sessions"

# Written when a bound is crossed, holding the reason. Its presence is what
# stops the loop, so recovering is deleting it, which only I can do.
BREAKER = HOME / "tripped"

# Every prompt picked up and what it cost, which is the one record both the
# rate and the daily total are read from.
LEDGER = HOME / "spending.jsonl"

# What a single turn may spend. Passed to the agent, so it is enforced where
# the money is actually spent rather than noticed here afterwards.
PER_TURN_USD = float(os.environ.get("AGENT_WORK_TURN_USD", "5"))

# What an hour and a day may come to. Deliberately tight: I would rather walk
# home through the park not looking at my phone than explain a four-figure bill
# to IT, so a false trip is the cheap outcome and the wrong one is not.
PER_HOUR_USD = float(os.environ.get("AGENT_WORK_HOURLY_USD", "10"))
PER_DAY_USD = float(os.environ.get("AGENT_WORK_DAILY_USD", "20"))

HOUR = 3600.0

# A prompt older than this is never run. The loop having been down is not a
# reason to come back and do everything that was said while it was: there is no
# backfill, by construction, whatever the reason a prompt went unnoticed.
STALE = 600.0

# One a minute is the most I ever send. More than that is not me.
MOST_PER_MINUTE = 1
MINUTE = 60.0

DAY = 86_400.0


class Tripped(Exception):
    """A bound was crossed. Nothing runs again until it is cleared by hand."""


def guard() -> None:
    """Raise if the breaker is down. Called before a pass, and before a turn."""
    if BREAKER.exists():
        raise Tripped(BREAKER.read_text().strip() or "no reason recorded")


def trip(why: str) -> None:
    """Stop everything, and say why in the place that keeps it stopped."""
    HOME.mkdir(parents=True, exist_ok=True)
    BREAKER.write_text(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {why}\n")
    log.problem(f"TRIPPED: {why}")
    log.problem(f"Nothing will run until you clear it: rm {BREAKER}")


def reset() -> str:
    """Turn it back on, and say what it had tripped on."""
    why = BREAKER.read_text().strip() if BREAKER.exists() else ""
    BREAKER.unlink(missing_ok=True)
    return why


def too_old(created: float) -> bool:
    """Whether a prompt is old enough that running it would be a backfill."""
    return created > 0 and (time.time() - created) > STALE


def note(session_id: str, prompt_id: int = 0, made: float = 0.0, usd: float = 0.0) -> None:
    """Record a prompt and what it cost. One file, and both questions read it.

    `made` is when I wrote the comment, not when this got to it. A loop that
    was busy for ten minutes then works through what arrived is not me typing
    quickly, and timing the pickup instead would trip on exactly that.
    """
    HOME.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("ab") as f:
        f.write(
            orjson.dumps(
                {
                    "at": time.time(),
                    "session": session_id,
                    "prompt": prompt_id,
                    "made": made or time.time(),
                    "usd": usd,
                }
            )
            + b"\n"
        )


def check_rate(session_id: str) -> None:
    """More than one comment a minute, on one session, is not something I did.

    Counted over when the comments were written and by distinct comment, so
    neither a slow pass nor the same prompt being looked at twice is mistaken
    for me typing faster than I ever do.
    """
    mine = [r for r in _since(DAY) if r.get("session") == session_id]
    recent = {r.get("prompt"): r for r in mine if time.time() - float(r.get("made", 0)) <= MINUTE}
    if len(recent) > MOST_PER_MINUTE:
        trip(f"{len(recent)} comments on {session_id} within a minute")
        raise Tripped(f"{len(recent)} comments in a minute")


def check_not_resubmitting(prompt_id: int) -> None:
    """A prompt reached twice is a bug in this, not a second request from me.

    Exact and unbounded rather than a cache of recent hashes: the ledger has
    every prompt ever run, one short line each, so "have I already done this"
    has a real answer instead of an approximate one. Keyed on the comment
    rather than on what it says, because saying "carry on" twice is a thing I
    genuinely do and running the same comment twice is never a thing I asked
    for. Every replay this thing has had would have stopped here.
    """
    if prompt_id and prompt_id in _submitted():
        trip(f"prompt {prompt_id} was already run once — resubmission is a bug")
        raise Tripped(f"prompt {prompt_id} already run")


def _submitted() -> set[int]:
    """Every prompt ever run. No window: "already done" does not expire."""
    return {int(r["prompt"]) for r in _all() if r.get("prompt")}


def check_spend() -> None:
    """The backstop, for whatever the other bounds did not think of.

    An hour as well as a day, because a runaway does its damage fast and a
    daily ceiling would let it run all afternoon before noticing.
    """
    for window, cap, name in ((HOUR, PER_HOUR_USD, "an hour"), (DAY, PER_DAY_USD, "a day")):
        spent = sum(float(r.get("usd") or 0) for r in _since(window))
        if spent > cap:
            trip(f"${spent:.2f} spent in {name}, over the ${cap:.0f} ceiling")
            raise Tripped(f"${spent:.2f} in {name}")


def spent_today() -> float:
    return sum(float(r.get("usd") or 0) for r in _since(DAY))


def _since(seconds: float) -> list[dict]:
    cutoff = time.time() - seconds
    return [r for r in _all() if r.get("at", 0) >= cutoff]


def _all() -> list[dict]:
    """The whole ledger. Two short lines a prompt, so this stays cheap for years."""
    if not LEDGER.exists():
        return []
    return [orjson.loads(line) for line in LEDGER.read_bytes().splitlines() if line.strip()]
