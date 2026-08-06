from pathlib import Path

from agent_sessions import skill
from agent_sessions.cli import main


def test_every_command_is_documented() -> None:
    """Generated from the command tree, so it cannot drift from the CLI."""
    text = skill.generate()
    for name in main.commands:
        assert f"### `agent-sessions {name}" in text


def test_commands_taking_an_id_say_so() -> None:
    text = skill.generate()
    assert "### `agent-sessions show ID`" in text
    assert "### `agent-sessions resume ID[@POINT]`" in text
    assert "### `agent-sessions search QUERY`" in text


def test_commands_taking_nothing_have_no_stray_metavar() -> None:
    assert "### `agent-sessions ls`" in skill.generate()


def test_meaningful_flags_are_listed() -> None:
    text = skill.generate()
    assert "`--tools`" in text
    assert "`-p, --project`" in text
    assert "`--fork`" in text


def test_universal_flags_are_left_out() -> None:
    """Repeating --json on every command would only cost the agent context."""
    text = skill.generate()
    assert "- `--json`" not in text
    assert "- `-h, --help`" not in text


def test_the_things_an_agent_would_otherwise_get_wrong_are_stated() -> None:
    text = skill.generate()
    assert "agent-sessions sync" in text
    assert "not indexed" in text
    assert "Exit codes" in text


def test_it_carries_frontmatter_a_skill_loader_can_read() -> None:
    text = skill.generate()
    assert text.startswith("---\nname: agent-sessions\n")
    assert text.count("---") >= 2


def test_install_writes_the_skill(tmp_path: Path) -> None:
    path = skill.install(tmp_path / "agent-sessions")
    assert path.name == "SKILL.md"
    assert path.read_text() == skill.generate()


def test_install_is_idempotent(tmp_path: Path) -> None:
    first = skill.install(tmp_path / "agent-sessions")
    second = skill.install(tmp_path / "agent-sessions")
    assert first == second


def test_examples_are_carried_through(tmp_path: Path) -> None:
    text = skill.generate()
    assert "$ agent-sessions search compaction" in text
    assert "```" in text


def test_example_indentation_survives() -> None:
    """A comment continued onto the next line must stay under the one it follows."""
    lines = skill.generate().splitlines()
    continuation = next(line for line in lines if line.lstrip().startswith("# matches Compaction"))
    assert continuation.startswith("    ")
