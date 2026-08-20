"""What the loop says about itself, for the times it is not doing what I expect.

It runs unattended for hours, so the questions it has to answer from a terminal
I come back to are: is it alive, did it see what I sent, what is it doing now,
and what went wrong. Nothing goes to stdout — that is for data.
"""

import pytest

from agent_sessions import log


def test_it_says_what_happened_on_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    """stdout is for data. This is not data."""
    log.say("picked up #4")
    captured = capsys.readouterr()
    assert "picked up #4" in captured.err
    assert captured.out == ""


def test_every_line_is_stamped(capsys: pytest.CaptureFixture[str]) -> None:
    """`it hung at some point` is not a bug report I can act on."""
    log.say("picked up #4")
    assert log.stamp()[:2] in capsys.readouterr().err


def test_the_detail_is_off_unless_asked_for(capsys: pytest.CaptureFixture[str]) -> None:
    """A poll every five seconds would bury everything worth reading."""
    log.detail("polled, nothing new")
    assert capsys.readouterr().err == ""


def test_asking_for_the_detail_gets_it(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(log, "VERBOSE", True)
    log.detail("polled, nothing new")
    assert "polled, nothing new" in capsys.readouterr().err


def test_a_problem_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    """Findable by eye in a wall of ordinary lines."""
    log.problem("github said 502")
    err = capsys.readouterr().err
    assert "github said 502" in err
    assert log.PROBLEM in err
