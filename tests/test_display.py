"""How one value reads. Cutting a turn down to size is the part with a decision in it."""

from agent_sessions import display

LINES = [f"line {i}" for i in range(400)]


def test_a_turn_worth_reading_is_left_alone() -> None:
    cut = display.trimmed("why is conform relocating worktrees")
    assert cut.head == "why is conform relocating worktrees"
    assert cut.rest == ""


def test_a_wall_of_lines_is_cut_at_a_line() -> None:
    cut = display.trimmed("\n".join(LINES))
    assert cut.head.splitlines() == LINES[: display.SHOWN_LINES]
    assert cut.rest.splitlines() == LINES[display.SHOWN_LINES :]
    assert cut.lines == 375


def test_nothing_is_lost_in_the_cutting() -> None:
    text = "here is the log\n" + "\n".join(LINES)
    cut = display.trimmed(text)
    assert cut.head + cut.rest == text


def test_one_long_line_is_cut_within_it() -> None:
    """A blob pasted as a single line has nowhere better to be cut."""
    cut = display.trimmed("x" * 8_000)
    assert len(cut.head) == display.SHOWN_CHARS
    assert len(cut.rest) == 6_000


def test_a_cut_inside_a_fence_closes_it_and_opens_it_again() -> None:
    text = "```python\n" + "\n".join(LINES) + "\n```"
    cut = display.trimmed(text)
    assert cut.head.endswith("\n```")
    assert cut.rest.startswith("```python\n")


def test_the_fence_it_reopens_is_not_counted_as_a_line_of_mine() -> None:
    cut = display.trimmed("```\n" + "\n".join(LINES) + "\n```")
    assert cut.lines == len(cut.rest.splitlines()) - 1
