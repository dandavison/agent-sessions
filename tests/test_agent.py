"""Starting an agent that already knows this tool."""

import os
from pathlib import Path

import pytest

from agent_sessions import agents, cli, skill


@pytest.fixture
def started(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Catch the handover, so the test process survives it."""
    calls: list[list[str]] = []
    monkeypatch.setattr(skill, "SKILLS_DIR", tmp_path / "agent-sessions")
    monkeypatch.setattr(agents.shutil, "which", lambda program: f"/usr/local/bin/{program}")
    monkeypatch.setattr(os, "execvp", lambda file, args: calls.append(list(args)))
    return calls


@pytest.fixture
def run(capsys: pytest.CaptureFixture[str]):
    def invoke(*args: str) -> tuple[int, str, str]:
        capsys.readouterr()
        code = cli.run(list(args))
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return invoke


def test_pi_on_the_default_model_is_what_you_get(started, run) -> None:
    run("agent")
    assert started[0][:3] == ["pi", "--model", agents.DEFAULT_MODELS["pi"]]


def test_qwen_can_be_asked_for(started, run) -> None:
    """Its own settings name a model already, so agent-sessions does not name one."""
    run("agent", "--with", "qwen")
    assert started[0][0] == "qwen"
    assert "--model" not in started[0]


def test_the_model_can_be_named(started, run) -> None:
    run("agent", "--model", "anthropic/claude-opus-5")
    run("agent", "--with", "qwen", "--model", "anthropic/claude-opus-5")
    assert started[0][:3] == ["pi", "--model", "anthropic/claude-opus-5"]
    assert started[1][:3] == ["qwen", "--model", "anthropic/claude-opus-5"]


# --- the skill travels in the opening message ------------------------------


def test_the_whole_skill_is_in_the_opening_message(started, run) -> None:
    """On disk it is only discoverable; here it is in context from the first turn."""
    run("agent")
    assert skill.body() in started[0][-1]


def test_the_frontmatter_is_left_behind(started, run) -> None:
    """It tells a loader when to reach for the skill. The agent is already reading it."""
    run("agent")
    assert "description: >-" not in started[0][-1]


def test_the_reference_is_fenced_off_from_the_task(started, run) -> None:
    message = (run("agent"), started[0][-1])[1]
    assert message.index("<reference>") < message.index("</reference>")
    assert message.rstrip().endswith(agents.OPENING_TASK)


def test_the_opening_task_asks_for_something_worth_seeing(started, run) -> None:
    run("agent")
    assert agents.OPENING_TASK in started[0][-1]


def test_a_task_of_my_own_replaces_it(started, run) -> None:
    run("agent", "what did I decide about compaction?")
    message = started[0][-1]
    assert message.rstrip().endswith("what did I decide about compaction?")
    assert agents.OPENING_TASK not in message
    assert skill.body() in message


def test_the_message_goes_where_each_agent_expects_it(started, run) -> None:
    """pi reads trailing words as the first message; for qwen that is -i."""
    run("agent")
    run("agent", "--with", "qwen")
    assert started[0][-2] != "--prompt-interactive"
    assert started[1][-2] == "--prompt-interactive"


def test_the_skill_is_still_written_out(started, run, tmp_path: Path) -> None:
    """Other agents find it there, and it is what the opening message quotes."""
    run("agent")
    assert (tmp_path / "agent-sessions" / "SKILL.md").read_text() == skill.generate()


def test_an_agent_that_is_not_installed_is_said_so(tmp_path, monkeypatch, run) -> None:
    monkeypatch.setattr(skill, "SKILLS_DIR", tmp_path / "agent-sessions")
    monkeypatch.setattr(agents.shutil, "which", lambda program: None)
    code, _, err = run("agent", "--with", "qwen")
    assert code == cli.EXIT_USAGE
    assert "qwen" in err


def test_an_unknown_agent_is_refused(started, run) -> None:
    code, _, _ = run("agent", "--with", "emacs")
    assert code == cli.EXIT_USAGE
    assert started == []
