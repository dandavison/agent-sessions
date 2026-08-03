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
    assert started == [["pi", "--model", agents.DEFAULT_MODELS["pi"]]]


def test_qwen_can_be_asked_for(started, run) -> None:
    """Its own settings name a model already, so agent-sessions does not name one."""
    run("agent", "--with", "qwen")
    assert started == [["qwen"]]


def test_the_model_can_be_named(started, run) -> None:
    run("agent", "--model", "anthropic/claude-opus-5")
    run("agent", "--with", "qwen", "--model", "anthropic/claude-opus-5")
    assert started[0] == ["pi", "--model", "anthropic/claude-opus-5"]
    assert started[1] == ["qwen", "--model", "anthropic/claude-opus-5"]


def test_a_prompt_goes_where_each_agent_expects_it(started, run) -> None:
    """pi reads trailing words as the first message; for qwen that is -i."""
    run("agent", "what did I decide about compaction?")
    run("agent", "--with", "qwen", "what did I decide about compaction?")
    assert started[0][-1] == "what did I decide about compaction?"
    assert started[1][-2:] == ["--prompt-interactive", "what did I decide about compaction?"]


def test_the_skill_is_written_before_the_handover(started, run, tmp_path: Path) -> None:
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
