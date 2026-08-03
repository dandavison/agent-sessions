"""Starting a coding agent that already knows this tool.

Discovery alone would put only the skill's description in context, and only
until something reminded the agent to read the rest. Appending it to the system
prompt puts the whole thing there from turn one.
"""

import os
import shutil
from pathlib import Path

from agent_sessions import skill

# pi has no default model of its own, so the one it is pointed at is named
# here, as provider/id; qwen takes its model from its own settings.
DEFAULT_MODELS: dict[str, str] = {
    "pi": "mlx/mlx-community/Qwen3.6-35B-A3B-4bit",
    "qwen": "",
}


class NotInstalled(Exception):
    def __init__(self, program: str) -> None:
        super().__init__(f"{program} is not on PATH.")


def start(program: str, model: str = "", prompt: str = "") -> None:
    """Replace this process with the agent, in the current directory."""
    if not shutil.which(program):
        raise NotInstalled(program)
    os.execvp(program, _command(program, model, prompt, skill.install(skill.SKILLS_DIR)))


def _command(program: str, model: str, prompt: str, skill_path: Path) -> list[str]:
    """Where each agent differs: pi takes the skill as a file, qwen as text.

    A prompt is pi's trailing words, and for qwen -i — not -p, which answers
    and exits rather than opening a session.
    """
    command = [program]
    if model := model or DEFAULT_MODELS[program]:
        command += ["--model", model]
    if program == "pi":
        command += ["--append-system-prompt", str(skill_path)]
    else:
        # One argument: given two, qwen reads the skill's leading --- as flags.
        command += [f"--append-system-prompt={skill_path.read_text()}"]
    if prompt:
        command += [prompt] if program == "pi" else ["--prompt-interactive", prompt]
    return command
