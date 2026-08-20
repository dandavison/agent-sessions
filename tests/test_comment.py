"""A turn as a GitHub comment, which is where the reading happens now.

GitHub renders markdown with syntax highlighting and collapses `<details>`, so
the two views asked for come free if the turn is laid out for them: prose at
the top level, every tool call folded away under it. The third view — just my
turns, without any of the agent's — is not something a comment can offer, since
GitHub has no collapse-all, so the issue body carries it instead.

Tool output goes to a third party the moment it is posted, and it holds things
that would never be committed. Redaction is a filter over everything on its way
out, not a thing each caller remembers.
"""

from agent_sessions import comment


def assistant(*blocks: dict) -> dict:
    return {"type": "assistant", "message": {"content": list(blocks)}}


def text(s: str) -> dict:
    return {"type": "text", "text": s}


def thinking(s: str) -> dict:
    return {"type": "thinking", "thinking": s}


def tool_use(name: str, **input: object) -> dict:
    return {"type": "tool_use", "id": "toolu_1", "name": name, "input": input}


def tool_result(content: object, is_error: bool = False) -> dict:
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": content,
                    "is_error": is_error,
                }
            ]
        },
    }


DONE = {"type": "result", "subtype": "success", "num_turns": 2, "total_cost_usd": 0.03}


# --- what is said comes first, and what was run folds away -------------------


def test_what_the_agent_said_is_at_the_top_level() -> None:
    """Read on a phone, from a notification: the answer should not need a tap."""
    out = comment.render([assistant(text("Replaced the retry loop.")), DONE])
    assert "Replaced the retry loop." in out
    assert "<details" not in out.split("Replaced the retry loop.")[0]


def test_a_tool_call_is_folded_away() -> None:
    out = comment.render(
        [assistant(tool_use("Bash", command="echo hello")), tool_result("hello"), DONE]
    )
    assert "<details>" in out
    assert "echo hello" in out
    assert "hello" in out


def test_the_fold_says_what_was_run_without_being_opened() -> None:
    """A summary that reads `Bash` and nothing else is a summary worth nothing."""
    out = comment.render([assistant(tool_use("Bash", command="uv run pytest -q")), DONE])
    summary = out.split("<summary>")[1].split("</summary>")[0]
    assert "Bash" in summary
    assert "uv run pytest -q" in summary


def test_thinking_is_not_posted() -> None:
    """It is long, it is not the answer, and it reads badly in a notification."""
    out = comment.render([assistant(thinking("Let me consider..."), text("Done.")), DONE])
    assert "Let me consider" not in out


# --- highlighting, which is the reason for using GitHub at all ---------------


def test_a_shell_command_is_fenced_as_shell() -> None:
    out = comment.render([assistant(tool_use("Bash", command="ls -la")), DONE])
    assert "```sh" in out


def test_an_edit_is_fenced_by_the_language_of_the_file() -> None:
    out = comment.render(
        [assistant(tool_use("Edit", file_path="/x/thing.py", new_string="x = 1")), DONE]
    )
    assert "```python" in out


def test_a_tool_with_no_language_of_its_own_is_fenced_as_json() -> None:
    out = comment.render([assistant(tool_use("Grep", pattern="def ", glob="*.py")), DONE])
    assert "```json" in out


# --- what must not leave this machine ----------------------------------------


def test_an_api_key_in_tool_output_is_not_posted() -> None:
    """The whole risk of the design in one test: output holds what commits never do."""
    leaked = "export ANTHROPIC_API_KEY=sk-ant-api03-" + "A" * 95
    out = comment.render(
        [assistant(tool_use("Bash", command="cat secret.sh")), tool_result(leaked)]
    )
    assert "sk-ant-api03-" not in out
    assert comment.REDACTED in out


def test_a_github_token_is_not_posted() -> None:
    out = comment.render(
        [assistant(tool_use("Bash", command="gh auth token")), tool_result("gho_" + "b" * 36)]
    )
    assert "gho_" not in out


def test_redaction_reaches_what_the_agent_says_too() -> None:
    """It reads files; it can repeat what it read."""
    out = comment.render([assistant(text("The key is sk-ant-api03-" + "C" * 95))])
    assert "sk-ant-api03-" not in out


# --- a comment GitHub will accept --------------------------------------------


def test_a_huge_tool_result_does_not_make_a_comment_github_refuses() -> None:
    """65,536 characters, and a test run clears that on its own."""
    out = comment.render(
        [assistant(tool_use("Bash", command="pytest")), tool_result("x" * 200_000), DONE]
    )
    assert len(out) <= comment.LIMIT


def test_truncation_says_that_it_truncated() -> None:
    out = comment.render(
        [assistant(tool_use("Bash", command="pytest")), tool_result("x" * 200_000), DONE]
    )
    assert "truncated" in out.lower()


# --- the issue body, which is the view a comment cannot give ------------------


def test_the_body_carries_the_frontmatter_needed_to_pick_the_session_up() -> None:
    session = {"id": "claude:7e90a7c6", "project": "wormhole", "cwd": "/Users/dan/src/wormhole"}
    out = comment.body(session, [])
    assert "claude:7e90a7c6" in out
    assert "wormhole" in out
    assert "/Users/dan/src/wormhole" in out


def test_the_body_is_a_table_so_a_phone_can_read_it() -> None:
    out = comment.body({"id": "claude:7e90a7c6"}, [])
    assert "|" in out and "---" in out


def test_the_body_holds_just_my_turns() -> None:
    """The view GitHub cannot give in the timeline: mine, without any of the agent's."""
    out = comment.body({"id": "claude:7e90a7c6"}, ["why is conform relocating", "try it with -x"])
    assert "why is conform relocating" in out
    assert "try it with -x" in out


def test_the_frontmatter_survives_a_round_trip() -> None:
    """It is read back off the issue to know which session a comment belongs to."""
    session = {"id": "claude:7e90a7c6", "project": "wormhole", "cwd": "/tmp/x"}
    assert comment.session_id(comment.body(session, [])) == "claude:7e90a7c6"
