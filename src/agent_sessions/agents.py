"""Starting a coding agent that already knows this tool.

The skill goes in the opening message rather than being left on disk to be
discovered, so the whole of it is in context from the first turn instead of
just the one-line description that draws an agent to read the rest.

Neither agent can pre-fill its input without also submitting it — pi takes
trailing words as the first message, qwen -i executes and then stays
interactive — so the opening message carries a task as well.
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


# Something worth seeing, and a demonstration of the tool in one.
OPENING_TASK = "Show me a table of my recent agent sessions."


def start(program: str, model: str = "", prompt: str = "") -> None:
    """Replace this process with the agent, in the current directory."""
    if not shutil.which(program):
        raise NotInstalled(program)
    skill.install(skill.SKILLS_DIR)
    os.execvp(program, _command(program, model, prompt or OPENING_TASK))


def _command(program: str, model: str, task: str) -> list[str]:
    """pi reads trailing words as the first message; for qwen that is -i.

    Not qwen's -p, which answers and exits rather than opening a session.
    """
    command = [program]
    if model := model or DEFAULT_MODELS[program]:
        command += ["--model", model]
    message = opening_message(task)
    return command + ([message] if program == "pi" else ["--prompt-interactive", message])


def opening_message(task: str) -> str:
    """The skill, then the thing to do with it.

    Fenced, because otherwise the skill's own headings and examples read as
    instructions addressed to the agent rather than as a document about a tool.
    """
    return f"""Below is the reference for `{skill.COMMAND}`, a command on this machine \
for finding and resuming past sessions with coding agents.

<reference>
{skill.body()}
</reference>

{task}"""
