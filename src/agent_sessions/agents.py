"""Starting a coding agent that already knows this tool.

Both agents discover skills in ~/.agents/skills, so knowing senderos is a
matter of writing the skill out and then handing the terminal over.
"""

import os
import shutil

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
    skill.install(skill.SKILLS_DIR)
    os.execvp(program, _command(program, model, prompt))


def _command(program: str, model: str, prompt: str) -> list[str]:
    """pi reads trailing words as the first message; for qwen that is -i.

    Not qwen's -p, which answers and exits rather than opening a session.
    """
    command = [program]
    if model := model or DEFAULT_MODELS[program]:
        command += ["--model", model]
    if prompt:
        command += [prompt] if program == "pi" else ["--prompt-interactive", prompt]
    return command
