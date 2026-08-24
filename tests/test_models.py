"""Turns: the unit a reader asks for, derived from what the one parser produces."""

from agent_sessions.models import Ran, Said, turn_at, turns


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


def test_a_turn_is_what_i_asked_and_what_followed() -> None:
    """The unit both readers show, and the unit it has to be able to match back."""
    (turn,) = turns([asked("why?", "u1"), said("because"), Ran(tool="Read", input={})])
    assert turn.key == "u1"
    assert turn.asked == "why?"
    assert len(turn.blocks) == 2


def test_each_thing_i_asked_starts_a_new_turn() -> None:
    found = turns([asked("first", "u1"), said("one"), asked("second", "u2"), said("two")])
    assert [t.key for t in found] == ["u1", "u2"]
    assert [t.asked for t in found] == ["first", "second"]


def test_work_before_anything_i_asked_is_not_a_turn() -> None:
    """A window can open mid-session, and the tail of an earlier turn is not mine."""
    assert turns([said("trailing"), asked("mine", "u9")])[0].key == "u9"
    assert len(turns([said("trailing"), asked("mine", "u9")])) == 1
