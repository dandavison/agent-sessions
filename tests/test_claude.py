import os
from pathlib import Path
from typing import Any

import orjson
import pytest
from conftest import (
    ai_title,
    assistant,
    compact_boundary,
    compact_summary,
    custom_title,
    last_prompt,
    relocated,
    text_block,
    tool_result,
    tool_use,
    user,
)

from agent_sessions.models import Delta, Running, Said
from agent_sessions.sources.claude import ClaudeSource, blocks, parse
from agent_sessions.sources.claude import _read as read
from agent_sessions.sources.claude import live as claude_live
from agent_sessions.sources.claude import render as claude_render

SESSION = "7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"


def write(tmp_path: Path, records: list[dict[str, Any]], name: str = SESSION) -> Path:
    project = tmp_path / "-Users-dan-src-wormhole"
    project.mkdir(parents=True, exist_ok=True)
    path = project / f"{name}.jsonl"
    path.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in records))
    return path


def ingest(tmp_path: Path, records: list[dict[str, Any]], name: str = SESSION) -> Delta:
    delta = parse(write(tmp_path, records, name), records)
    assert delta is not None
    return delta


# A straight thread: two exchanges, no branching.
def linear() -> list[dict[str, Any]]:
    return [
        user("u1", None, "why is conform relocating worktrees"),
        assistant("a1", "u1", [text_block("Because it reconciles submodules.")], "req_1"),
        user("u2", "a1", "show me the code"),
        assistant("a2", "u2", [text_block("See git.rs.")], "req_2"),
        last_prompt("a2"),
    ]


def test_linear_transcript(tmp_path: Path) -> None:
    delta = ingest(tmp_path, linear())
    assert delta.session.id == f"claude:{SESSION}"
    assert delta.session.agent == "claude"
    assert delta.session.cwd == "/Users/dan/src/wormhole"
    assert delta.session.git_branch == "main"
    assert delta.session.model == "claude-opus-5"
    assert delta.session.n_messages == 4
    assert delta.session.n_user_turns == 2
    assert [n.text for n in delta.nodes if n.text] == [
        "why is conform relocating worktrees",
        "Because it reconciles submodules.",
        "show me the code",
        "See git.rs.",
    ]


def test_sidecar_only_transcript_is_not_a_session(tmp_path: Path) -> None:
    records = [ai_title("Started and abandoned"), last_prompt("nothing")]
    assert parse(write(tmp_path, records), records) is None


def test_timestamps_span_the_transcript(tmp_path: Path) -> None:
    s = ingest(tmp_path, linear()).session
    assert s.started_at is not None and s.ended_at is not None
    assert s.started_at <= s.ended_at


# --- text extraction -------------------------------------------------------


def test_tool_calls_and_results_are_not_indexed(tmp_path: Path) -> None:
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "list the files"),
            assistant("a1", "u1", [tool_use()], "req_1"),
            tool_result("u2", "a1"),
            assistant("a2", "u2", [text_block("Two files.")], "req_2"),
            last_prompt("a2"),
        ],
    )
    assert [n.text for n in delta.nodes if n.text] == ["list the files", "Two files."]


def test_thinking_is_not_indexed(tmp_path: Path) -> None:
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "think about it"),
            assistant("a1", "u1", [{"type": "thinking", "thinking": "hmm, secret"}], "req_1"),
            assistant("a2", "a1", [text_block("Done.")], "req_1"),
        ],
    )
    assert [n.text for n in delta.nodes if n.text] == ["think about it", "Done."]


def test_string_content_is_indexed(tmp_path: Path) -> None:
    record = user("u1", None, "x")
    record["message"]["content"] = "a plain string prompt"
    delta = ingest(tmp_path, [record])
    assert [n.text for n in delta.nodes if n.text] == ["a plain string prompt"]


def test_meta_records_are_not_indexed(tmp_path: Path) -> None:
    delta = ingest(
        tmp_path,
        [user("u1", None, "[Image: source]", isMeta=True), user("u2", "u1", "the real prompt")],
    )
    assert [n.text for n in delta.nodes if n.text] == ["the real prompt"]


def test_user_turn_count_ignores_tool_results(tmp_path: Path) -> None:
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "go"),
            assistant("a1", "u1", [tool_use()], "req_1"),
            tool_result("u2", "a1"),
            assistant("a2", "u2", [text_block("done")], "req_2"),
            last_prompt("a2"),
        ],
    )
    assert delta.session.n_user_turns == 1


# --- branching -------------------------------------------------------------


# u1 was answered, then rewound: a1 has two children, only one of them live.
def branched() -> list[dict[str, Any]]:
    return [
        user("u1", None, "first question"),
        assistant("a1", "u1", [text_block("first answer")], "req_1"),
        user("u2", "a1", "abandoned question"),
        assistant("a2", "u2", [text_block("abandoned answer")], "req_2"),
        user("u3", "a1", "live question"),
        assistant("a3", "u3", [text_block("live answer")], "req_3"),
        last_prompt("a3"),
    ]


def test_branch_point_is_marked(tmp_path: Path) -> None:
    nodes = {n.uuid: n for n in ingest(tmp_path, branched()).nodes}
    assert nodes["a1"].is_branch_point
    assert not nodes["u1"].is_branch_point
    assert not nodes["a3"].is_branch_point


def test_abandoned_branches_are_still_searchable(tmp_path: Path) -> None:
    texts = [n.text for n in ingest(tmp_path, branched()).nodes if n.text]
    assert "abandoned question" in texts


def test_turn_count_includes_abandoned_branches(tmp_path: Path) -> None:
    """Everything I typed, I typed — even the attempt I then rewound."""
    assert ingest(tmp_path, branched()).session.n_user_turns == 3


def test_leaf_selects_which_branch_is_live(tmp_path: Path) -> None:
    records = branched()
    records[-1] = last_prompt("a2")
    assert ingest(tmp_path, records).session.context_tokens == 1102


def test_without_last_prompt_the_final_record_is_the_leaf(tmp_path: Path) -> None:
    """The live thread ends at a3, so its context is the one reported."""
    assert ingest(tmp_path, branched()[:-1]).session.context_tokens == 1102


def test_leaf_naming_a_missing_record_is_ignored(tmp_path: Path) -> None:
    records = [*branched()[:-1], last_prompt("no-such-uuid")]
    assert ingest(tmp_path, records).session.context_tokens == 1102


# --- compaction ------------------------------------------------------------


def compacted() -> list[dict[str, Any]]:
    return [
        user("u1", None, "a long piece of work"),
        assistant("a1", "u1", [text_block("ok")], "req_1"),
        compact_boundary("c1", logical_parent="a1", anchor="s1"),
        compact_summary("s1", "c1", "Summary: the earlier work concerned worktrees."),
        user("u2", "s1", "carry on"),
        assistant("a2", "u2", [text_block("carrying on")], "req_2"),
        last_prompt("a2"),
    ]


def test_compaction_is_recorded(tmp_path: Path) -> None:
    (compaction,) = ingest(tmp_path, compacted()).compactions
    assert compaction.trigger == "auto"
    assert compaction.pre_tokens == 1_000_823
    assert compaction.post_tokens == 22_191
    assert compaction.logical_parent_uuid == "a1"
    assert compaction.anchor_uuid == "s1"
    assert compaction.preserved_count == 3


def test_dropped_tokens_come_from_the_boundary(tmp_path: Path) -> None:
    assert ingest(tmp_path, compacted()).session.dropped_tokens == 1_000_823 - 22_191


def test_compact_summary_is_indexed_as_its_own_role(tmp_path: Path) -> None:
    nodes = {n.uuid: n for n in ingest(tmp_path, compacted()).nodes}
    assert nodes["s1"].role == "summary"
    assert "worktrees" in nodes["s1"].text


def test_pre_compaction_turns_survive_the_new_root(tmp_path: Path) -> None:
    """The boundary has parentUuid null, so the live walk cannot reach behind it."""
    delta = ingest(tmp_path, compacted())
    assert "a long piece of work" in [n.text for n in delta.nodes if n.text]
    assert delta.session.n_user_turns == 2


def test_two_compactions_are_both_recorded(tmp_path: Path) -> None:
    records = [
        *compacted()[:-1],
        compact_boundary(
            "c2", logical_parent="a2", anchor="s2", pre=900_000, post=10_000, trigger="manual"
        ),
        compact_summary("s2", "c2", "Second summary."),
        last_prompt("s2"),
    ]
    delta = ingest(tmp_path, records)
    assert [c.trigger for c in delta.compactions] == ["auto", "manual"]
    assert delta.session.dropped_tokens == 1_000_823 - 22_191


# --- tokens ----------------------------------------------------------------


def test_context_is_the_latest_value_not_a_sum(tmp_path: Path) -> None:
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "one"),
            assistant(
                "a1",
                "u1",
                [text_block("x")],
                "req_1",
                usage={
                    "input_tokens": 1,
                    "cache_read_input_tokens": 100,
                    "cache_creation_input_tokens": 10,
                    "output_tokens": 5,
                },
            ),
            user("u2", "a1", "two"),
            assistant(
                "a2",
                "u2",
                [text_block("y")],
                "req_2",
                usage={
                    "input_tokens": 2,
                    "cache_read_input_tokens": 999_154,
                    "cache_creation_input_tokens": 628,
                    "output_tokens": 7,
                },
            ),
            last_prompt("a2"),
        ],
    )
    assert delta.session.context_tokens == 2 + 999_154 + 628


def test_output_tokens_dedupe_by_request_id(tmp_path: Path) -> None:
    """One response is split across records that repeat a single usage object."""
    usage = {
        "input_tokens": 1,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 1000,
    }
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "go"),
            assistant("a1", "u1", [{"type": "thinking", "thinking": "…"}], "req_1", usage=usage),
            assistant("a2", "a1", [text_block("part")], "req_1", usage=usage),
            assistant("a3", "a2", [tool_use()], "req_1", usage=usage),
            last_prompt("a3"),
        ],
    )
    assert delta.session.output_tokens == 1000


def test_output_tokens_sum_across_distinct_requests(tmp_path: Path) -> None:
    usage = {
        "input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 10,
    }
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "go"),
            assistant("a1", "u1", [text_block("one")], "req_1", usage=usage),
            assistant("a2", "a1", [text_block("two")], "req_2", usage=usage),
            last_prompt("a2"),
        ],
    )
    assert delta.session.output_tokens == 20


def test_context_ignores_the_abandoned_branch(tmp_path: Path) -> None:
    records = branched()
    records[3] = assistant(
        "a2",
        "u2",
        [text_block("abandoned answer")],
        "req_2",
        usage={
            "input_tokens": 0,
            "cache_read_input_tokens": 999_999,
            "cache_creation_input_tokens": 0,
            "output_tokens": 1,
        },
    )
    assert ingest(tmp_path, records).session.context_tokens == 1102


# --- titles, cwd, forks ----------------------------------------------------


def test_custom_title_beats_ai_title(tmp_path: Path) -> None:
    records = [*linear(), ai_title("generated"), custom_title("mine")]
    assert ingest(tmp_path, records).session.title == "mine"


def test_ai_title_beats_the_first_prompt(tmp_path: Path) -> None:
    assert ingest(tmp_path, [*linear(), ai_title("generated")]).session.title == "generated"


def test_slug_is_used_when_there_is_no_title(tmp_path: Path) -> None:
    records = linear()
    records[3]["slug"] = "pure-treehouse"
    assert ingest(tmp_path, records).session.title == "pure-treehouse"


def test_first_prompt_is_the_last_resort_title(tmp_path: Path) -> None:
    assert ingest(tmp_path, linear()).session.title == "why is conform relocating worktrees"


def test_relocation_moves_the_cwd(tmp_path: Path) -> None:
    records = [*linear(), relocated("/Users/dan/src/tide")]
    assert ingest(tmp_path, records).session.cwd == "/Users/dan/src/tide"


def test_fork_edge_points_at_the_splice(tmp_path: Path) -> None:
    fork = {"sessionId": "373dfbf8-parent", "messageUuid": "f39f475e"}
    records = [
        user("f39f475e", None, "picked up from elsewhere", forkedFrom=fork),
        assistant("a1", "f39f475e", [text_block("ok")], "req_1", forkedFrom=fork),
    ]
    (edge,) = ingest(tmp_path, records).edges
    assert edge.parent == "claude:373dfbf8-parent"
    assert edge.kind == "fork"
    assert edge.at_uuid == "f39f475e"


def test_unforked_transcript_has_no_edges(tmp_path: Path) -> None:
    assert ingest(tmp_path, linear()).edges == []


# --- where a session can be resumed from -----------------------------------


def test_a_session_is_found_from_the_directory_it_was_had_in(tmp_path: Path) -> None:
    path = write(tmp_path, linear())
    assert ClaudeSource().resumable_from(path, "/Users/dan/src/wormhole")
    assert not ClaudeSource().resumable_from(path, "/Users/dan/src/wormhole/gui")
    assert not ClaudeSource().resumable_from(path, "/Users/dan/worktrees/wormhole/x/wormhole")


def test_a_dot_in_the_path_is_a_dash_in_the_directory_name(tmp_path: Path) -> None:
    """Submodule worktrees live under `.git`, and 189 of my sessions were had in one."""
    project = tmp_path / "-Users-dan-src-devenv--git-modules-dotfiles"
    project.mkdir()
    path = project / f"{SESSION}.jsonl"
    path.touch()
    assert ClaudeSource().resumable_from(path, "/Users/dan/src/devenv/.git/modules/dotfiles")


# --- resuming at a point ---------------------------------------------------


def forked_at(tmp_path: Path, records: list[dict[str, Any]], at: str) -> tuple[str, list[dict]]:
    path = write(tmp_path, records)
    native = ClaudeSource(tmp_path).fork_at(path, at)
    written = path.with_name(f"{native}.jsonl")
    return native, [orjson.loads(line) for line in written.read_bytes().splitlines() if line]


def test_a_point_is_resumed_by_writing_a_session_that_ends_there(tmp_path: Path) -> None:
    """Claude can only resume at a leaf, so the point has to become one."""
    _, records = forked_at(tmp_path, branched(), "a1")
    assert [r["uuid"] for r in records] == ["u1", "a1"]


def test_the_branch_the_point_is_on_is_the_one_kept(tmp_path: Path) -> None:
    _, records = forked_at(tmp_path, branched(), "a2")
    assert [r["uuid"] for r in records] == ["u1", "a1", "u2", "a2"]


def test_what_was_written_is_a_session_of_its_own(tmp_path: Path) -> None:
    native, records = forked_at(tmp_path, branched(), "a1")
    assert {r["sessionId"] for r in records} == {native}
    assert native != SESSION


def test_each_record_says_where_it_came_from(tmp_path: Path) -> None:
    """A fork is stamped record by record, exactly as Claude stamps its own."""
    _, records = forked_at(tmp_path, branched(), "a1")
    assert all(r["forkedFrom"] == {"sessionId": SESSION, "messageUuid": r["uuid"]} for r in records)


def test_the_new_session_indexes_as_a_fork_of_the_old(tmp_path: Path) -> None:
    native, records = forked_at(tmp_path, branched(), "a1")
    delta = parse(tmp_path / "-Users-dan-src-wormhole" / f"{native}.jsonl", records)
    assert delta is not None
    (edge,) = delta.edges
    assert edge.child == f"claude:{native}" and edge.parent == f"claude:{SESSION}"


def test_a_compaction_summary_is_carried_over(tmp_path: Path) -> None:
    """Resuming after a compaction needs the boundary and the summary, not the era before it."""
    _, records = forked_at(tmp_path, compacted(), "u2")
    assert [r["uuid"] for r in records] == ["c1", "s1", "u2"]


def test_a_point_that_is_not_in_the_transcript_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        forked_at(tmp_path, linear(), "no-such-uuid")


# --- discovery -------------------------------------------------------------


def test_discover_finds_sessions_but_not_subagents(tmp_path: Path) -> None:
    """Nobody talked to a subagent, so its transcript is not a session."""
    path = write(tmp_path, linear())
    subagents = path.parent / SESSION / "subagents"
    subagents.mkdir(parents=True)
    (subagents / "agent-abc123.jsonl").write_bytes(
        orjson.dumps(user("s1", None, "explore this", agentId="abc123", isSidechain=True)) + b"\n"
    )

    assert {d.id for d in ClaudeSource(tmp_path).discover()} == {f"claude:{SESSION}"}


def test_discovery_does_not_escape_the_projects_dir(tmp_path: Path) -> None:
    """`projects.bak-*` snapshots sit beside `projects/`, holding another 382 MB."""
    projects = tmp_path / "projects"
    write(projects, linear())
    backup = tmp_path / "projects.bak-20260731-101422" / "-Users-dan-src-wormhole"
    backup.mkdir(parents=True)
    (backup / "old.jsonl").write_bytes(orjson.dumps(user("u1", None, "old")) + b"\n")

    assert len(ClaudeSource(projects).discover()) == 1


# --- interruptions and multiple roots --------------------------------------


def test_flagged_interrupt_is_not_my_turn(tmp_path: Path) -> None:
    """Hitting escape writes a synthetic user record. Nobody typed it."""
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "do the thing"),
            assistant("a1", "u1", [text_block("starting")], "req_1"),
            user("u2", "a1", "[Request interrupted by user]", interruptedMessageId="msg_1"),
            user("u3", "u2", "actually, do this instead"),
            last_prompt("u3"),
        ],
    )
    assert [n.text for n in delta.nodes if n.text] == [
        "do the thing",
        "starting",
        "actually, do this instead",
    ]
    assert delta.session.n_user_turns == 2


def test_unflagged_interrupt_is_recognised_by_its_text(tmp_path: Path) -> None:
    """A tool-use interruption carries no flag at all; the sentinel string is all there is."""
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "do the thing"),
            user("u2", "u1", "[Request interrupted by user for tool use]"),
            last_prompt("u2"),
        ],
    )
    assert [n.text for n in delta.nodes if n.text] == ["do the thing"]
    assert delta.session.n_user_turns == 1


def test_an_interruption_i_actually_quoted_is_kept(tmp_path: Path) -> None:
    """Only the bare sentinel is synthetic; the same words inside a real prompt are mine."""
    delta = ingest(
        tmp_path,
        [user("u1", None, "why does [Request interrupted by user] keep appearing?")],
    )
    assert delta.session.n_user_turns == 1


def test_a_second_root_does_not_lose_earlier_turns(tmp_path: Path) -> None:
    """A transcript can hold several unconnected chains; all of them are mine."""
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "first chain"),
            assistant("a1", "u1", [text_block("ok")], "req_1"),
            user("u2", None, "second chain, unconnected"),
            assistant("a2", "u2", [text_block("ok again")], "req_2"),
            last_prompt("a2"),
        ],
    )
    assert delta.session.n_user_turns == 2
    assert "first chain" in [n.text for n in delta.nodes if n.text]


def test_the_whole_dag_is_kept_not_only_prose(tmp_path: Path) -> None:
    """Silent records are stored so the branch structure survives."""
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "go"),
            assistant("a1", "u1", [tool_use()], "req_1"),
            tool_result("u2", "a1"),
            last_prompt("u2"),
        ],
    )
    assert [n.uuid for n in delta.nodes] == ["u1", "a1", "u2"]
    assert [n.uuid for n in delta.nodes if n.text] == ["u1"]


def test_a_rewind_at_a_silent_node_is_still_a_branch(tmp_path: Path) -> None:
    """Most rewinds land on a tool call or a timing record, not on anything said."""
    delta = ingest(
        tmp_path,
        [
            user("u1", None, "go"),
            assistant("a1", "u1", [tool_use()], "req_1"),
            user("u2", "a1", "first attempt"),
            user("u3", "a1", "second attempt"),
            last_prompt("u3"),
        ],
    )
    assert {n.uuid for n in delta.nodes if n.is_branch_point} == {"a1"}


# --- rendering to markdown -------------------------------------------------


def rendered(records: list[dict[str, Any]], tools: bool = False, whole: bool = False) -> str:
    return "".join(claude_render(records, tools=tools, whole=whole))


def test_render_prose_as_markdown() -> None:
    out = rendered(linear())
    assert "## user\n\nwhy is conform relocating worktrees" in out
    assert "## assistant\n\nBecause it reconciles submodules." in out


def test_render_omits_tools_by_default() -> None:
    records = [
        user("u1", None, "list the files"),
        assistant("a1", "u1", [tool_use()], "req_1"),
        tool_result("u2", "a1"),
        last_prompt("u2"),
    ]
    assert "ls" not in rendered(records)


def test_render_with_tools_shows_the_call_and_its_output() -> None:
    """The index holds no tool output, so this is the only way to see what ran."""
    records = [
        user("u1", None, "list the files"),
        assistant("a1", "u1", [tool_use()], "req_1"),
        tool_result("u2", "a1"),
        last_prompt("u2"),
    ]
    out = rendered(records, tools=True)
    assert "### Bash" in out
    assert '"command": "ls"' in out
    assert "a\nb" in out


def test_render_follows_the_live_branch() -> None:
    assert "abandoned question" not in rendered(branched())


def test_render_whole_includes_abandoned_branches() -> None:
    assert "abandoned question" in rendered(branched(), whole=True)


def test_render_marks_a_compaction() -> None:
    out = rendered(compacted())
    assert "*Compacted (auto): 1,000,823 → 22,191 tokens*" in out


# --- the live session registry ---------------------------------------------


def write_registry(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for i, record in enumerate(records):
        (sessions / f"{i}.json").write_bytes(orjson.dumps(record))
    return sessions


def test_live_reports_running_sessions(tmp_path: Path) -> None:
    sessions = write_registry(tmp_path, [{"pid": os.getpid(), "sessionId": "s1", "status": "idle"}])
    assert claude_live(sessions) == {"s1": Running(pid=os.getpid(), status="idle")}


def test_live_carries_the_pid_so_its_pane_can_be_found(tmp_path: Path) -> None:
    """Focusing the pane a session is already in needs the process, not just the id."""
    sessions = write_registry(tmp_path, [{"pid": os.getpid(), "sessionId": "s1", "status": "busy"}])
    assert claude_live(sessions)["s1"].pid == os.getpid()


def test_a_dead_process_is_not_a_live_session(tmp_path: Path) -> None:
    """The files outlive the process that wrote them."""
    sessions = write_registry(tmp_path, [{"pid": 2**31 - 1, "sessionId": "s1", "status": "idle"}])
    assert claude_live(sessions) == {}


def test_registry_entry_without_a_pid_is_ignored(tmp_path: Path) -> None:
    sessions = write_registry(tmp_path, [{"sessionId": "s1", "status": "idle"}])
    assert claude_live(sessions) == {}


def test_no_registry_at_all(tmp_path: Path) -> None:
    assert claude_live(tmp_path / "nothing") == {}


# --- what a turn run with nobody watching is allowed to do -------------------


def test_a_turn_is_resumed_headlessly_and_asked_only_for_its_summary() -> None:
    """No allowlist and no setting-sources: a turn runs under my own settings.

    Deliberate, and worth a test so it is not quietly "fixed" back. Mine say
    `defaultMode: bypassPermissions`, so a prompt left on the control channel
    is allowed whatever a prompt typed at the keyboard is allowed. What stands
    between a comment and this machine is who can reach the repo, and nothing
    else.
    """
    assert ClaudeSource().turn_command("7e90")[:6] == [
        "claude",
        "-p",
        "--resume",
        "7e90",
        "--output-format",
        "json",
    ]


def test_a_turn_is_given_a_budget_it_cannot_exceed() -> None:
    """The only bound on a turn that runs away inside itself.

    Everything else here counts prompts; nothing else stops one turn spending
    without limit. The agent enforces this on itself, which is why it is a flag
    rather than something noticed afterwards from the summary.
    """
    command = ClaudeSource().turn_command("7e90")
    assert "--max-budget-usd" in command
    assert float(command[command.index("--max-budget-usd") + 1]) > 0


def test_a_transcript_being_written_to_can_still_be_read(tmp_path: Path) -> None:
    """A turn appends while I read, so the last line is sometimes half a line.

    Progress comes from watching the file grow during a turn, so this is the
    normal case, not a corrupt file.
    """
    path = tmp_path / "s.jsonl"
    whole = orjson.dumps(user("u1", None, "hello"))
    path.write_bytes(whole + b"\n" + whole[:20])
    assert [r["uuid"] for r in read(path)] == ["u1"]


def test_a_block_knows_where_it_came_from(tmp_path: Path) -> None:
    """A rendered turn has to be matched back to the turn it renders.

    Without an identity from the transcript, reconciling the thread means
    guessing by position or by text, and both drift.
    """
    records = [
        user("u1", None, "why is conform relocating"),
        assistant("a1", "u1", [text_block("Because of submodules.")], "req_1"),
        last_prompt("a1"),
    ]
    said = [b for b in blocks(records) if isinstance(b, Said)]
    assert [b.uuid for b in said] == ["u1", "a1"]


def test_a_turn_in_flight_is_invisible_from_the_recorded_leaf(tmp_path: Path) -> None:
    """Progress said `0 so far` for the whole turn, then everything at once.

    The live thread is walked back from the leaf the sidecar names, and that is
    not written until the turn ends. Everything the turn appends hangs below it
    and is not on the branch, so watching a turn this way sees nothing.
    """
    records = [
        user("u1", None, "the first thing"),
        assistant("a1", "u1", [text_block("the first answer")], "req_1"),
        last_prompt("a1"),
        user("u2", "a1", "asked while running"),
        assistant("a2", "u2", [text_block("answered while running")], "req_2"),
    ]
    settled = [b.text for b in blocks(records) if isinstance(b, Said)]
    assert "asked while running" not in settled

    growing = [b.text for b in blocks(records, tip=True) if isinstance(b, Said)]
    assert "asked while running" in growing
    assert "answered while running" in growing


def test_the_recorded_leaf_still_decides_once_nothing_is_running(tmp_path: Path) -> None:
    """It is what keeps a rewound session from reading as its abandoned branch."""
    records = [
        user("u1", None, "kept"),
        assistant("a1", "u1", [text_block("kept answer")], "req_1"),
        user("u2", "u1", "abandoned"),
        last_prompt("a1"),
    ]
    assert "abandoned" not in [b.text for b in blocks(records) if isinstance(b, Said)]
