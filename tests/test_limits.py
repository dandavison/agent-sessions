"""The bounds, each written against the thing that actually went wrong.

Every one of these is a real incident, not a hypothetical. Nine answered
prompts queued for replay after a compaction. A turn that could have run away
inside itself with nothing to stop it. A day that could quietly add up.

The point of a breaker is that it is dumb and cannot be talked out of it, so
these are the tests for exactly that: it trips, and nothing but a hand puts it
back.
"""

import time

import pytest

from agent_sessions import limits


def test_a_prompt_run_once_is_never_run_again() -> None:
    """The kill switch for every replay this thing has had.

    Exact, not a cache of recent hashes: the ledger holds every prompt ever
    run, so this has a real answer rather than an approximate one, however long
    ago the first run was.
    """
    limits.note("claude:7e90", prompt_id=555)
    with pytest.raises(limits.Tripped):
        limits.check_not_resubmitting(555)


def test_a_prompt_from_last_year_still_counts_as_run() -> None:
    """No window. "Have I already done this" does not expire."""
    limits.note("claude:7e90", prompt_id=556)
    _backdate(limits.DAY * 400)
    with pytest.raises(limits.Tripped):
        limits.check_not_resubmitting(556)


def test_a_prompt_never_run_is_allowed_through() -> None:
    limits.note("claude:7e90", prompt_id=1)
    limits.check_not_resubmitting(2)


def test_two_comments_in_a_minute_trips_it() -> None:
    """I do not type that fast, so it is not me, so nothing should run."""
    now = time.time()
    limits.note("claude:7e90", prompt_id=1, made=now - 5)
    limits.note("claude:7e90", prompt_id=2, made=now - 3)
    with pytest.raises(limits.Tripped):
        limits.check_rate("claude:7e90")


def test_two_comments_further_apart_are_fine() -> None:
    """A pass that was busy for ten minutes is not me typing quickly."""
    now = time.time()
    limits.note("claude:7e90", prompt_id=1, made=now - 300)
    limits.note("claude:7e90", prompt_id=2, made=now - 120)
    limits.check_rate("claude:7e90")


def test_another_session_does_not_count_against_this_one() -> None:
    now = time.time()
    limits.note("claude:aaaa", prompt_id=1, made=now - 5)
    limits.note("claude:bbbb", prompt_id=2, made=now - 3)
    limits.check_rate("claude:aaaa")


def test_a_day_over_the_ceiling_trips_it() -> None:
    limits.note("claude:7e90", prompt_id=1, usd=limits.PER_DAY_USD + 1)
    with pytest.raises(limits.Tripped):
        limits.check_spend()


def test_an_old_prompt_is_never_run() -> None:
    """No backfill, by construction. Being down is not a reason to catch up."""
    assert limits.too_old(time.time() - limits.STALE - 1)
    assert not limits.too_old(time.time() - 5)
    assert not limits.too_old(0), "a comment with no timestamp is not ancient"


def test_nothing_runs_once_it_has_tripped() -> None:
    limits.trip("because")
    with pytest.raises(limits.Tripped, match="because"):
        limits.guard()


def test_only_a_hand_puts_it_back() -> None:
    """Recovering automatically from "that spent too much" is the same bug twice."""
    limits.trip("because")
    assert "because" in limits.reset()
    limits.guard()


def _backdate(by: float) -> None:
    """Age every record, to ask what this looks like a long time later."""
    lines = limits.LEDGER.read_bytes().splitlines()
    import orjson

    limits.LEDGER.write_bytes(
        b"\n".join(
            orjson.dumps({**orjson.loads(x), "at": orjson.loads(x)["at"] - by}) for x in lines if x
        )
    )
