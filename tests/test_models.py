"""Turns: the unit a reader asks for, derived from what the one parser produces."""

from agent_sessions.models import Said, turn_at, turns


def asked(text: str, uuid: str) -> Said:
    return Said(role="user", text=text, uuid=uuid)


def said(text: str, uuid: str = "") -> Said:
    return Said(role="assistant", text=text, uuid=uuid)


def test_a_turn_can_be_had_back_with_the_prompt_that_started_it() -> None:
    """`blocks` is the answer alone; showing a turn means showing what was asked."""
    (turn,) = turns([asked("why?", "u1"), said("because", "a1")])
    assert turn.whole == [asked("why?", "u1"), said("because", "a1")]


def test_a_point_names_the_turn_it_falls_in() -> None:
    """`show --turns` prints prompts, `tree` prints replies; either may be handed back."""
    blocks = [asked("first", "u1"), said("one", "a1"), asked("second", "u2"), said("two", "a2")]
    assert (found := turn_at(blocks, "u2")) and found.asked == "second"
    assert (found := turn_at(blocks, "a1")) and found.asked == "first"


def test_a_point_inside_no_turn_names_no_turn() -> None:
    assert turn_at([asked("mine", "u1"), said("answer", "a1")], "a9") is None
