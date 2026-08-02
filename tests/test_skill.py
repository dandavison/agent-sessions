from pathlib import Path

from senderos import skill
from senderos.cli import main


def test_every_command_is_documented() -> None:
    """Generated from the command tree, so it cannot drift from the CLI."""
    text = skill.generate()
    for name in main.commands:
        assert f"### `senderos {name}" in text


def test_commands_taking_an_id_say_so() -> None:
    text = skill.generate()
    assert "### `senderos show ID`" in text
    assert "### `senderos resume ID`" in text
    assert "### `senderos search QUERY`" in text


def test_commands_taking_nothing_have_no_stray_metavar() -> None:
    assert "### `senderos ls`" in skill.generate()


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
    assert "senderos sync" in text
    assert "not indexed" in text
    assert "Exit codes" in text


def test_it_carries_frontmatter_a_skill_loader_can_read() -> None:
    text = skill.generate()
    assert text.startswith("---\nname: senderos\n")
    assert text.count("---") >= 2


def test_install_writes_the_skill(tmp_path: Path) -> None:
    path = skill.install(tmp_path / "senderos")
    assert path.name == "SKILL.md"
    assert path.read_text() == skill.generate()


def test_install_is_idempotent(tmp_path: Path) -> None:
    first = skill.install(tmp_path / "senderos")
    second = skill.install(tmp_path / "senderos")
    assert first == second
