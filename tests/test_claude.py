from pathlib import Path
from typing import Any

import orjson
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

from senderos.models import Delta
from senderos.sources.claude import ClaudeSource, parse

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
    assert delta.sendero.id == f"claude:{SESSION}"
    assert delta.sendero.agent == "claude"
    assert delta.sendero.cwd == "/Users/dan/src/wormhole"
    assert delta.sendero.git_branch == "main"
    assert delta.sendero.model == "claude-opus-5"
    assert delta.sendero.n_messages == 4
    assert delta.sendero.n_user_turns == 2
    assert [n.text for n in delta.nodes if n.text] == [
        "why is conform relocating worktrees",
        "Because it reconciles submodules.",
        "show me the code",
        "See git.rs.",
    ]


def test_sidecar_only_transcript_is_not_a_sendero(tmp_path: Path) -> None:
    records = [ai_title("Started and abandoned"), last_prompt("nothing")]
    assert parse(write(tmp_path, records), records) is None


def test_timestamps_span_the_transcript(tmp_path: Path) -> None:
    s = ingest(tmp_path, linear()).sendero
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
    assert delta.sendero.n_user_turns == 1


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
    assert ingest(tmp_path, branched()).sendero.n_user_turns == 3


def test_leaf_selects_which_branch_is_live(tmp_path: Path) -> None:
    records = branched()
    records[-1] = last_prompt("a2")
    assert ingest(tmp_path, records).sendero.context_tokens == 1102


def test_without_last_prompt_the_final_record_is_the_leaf(tmp_path: Path) -> None:
    """The live thread ends at a3, so its context is the one reported."""
    assert ingest(tmp_path, branched()[:-1]).sendero.context_tokens == 1102


def test_leaf_naming_a_missing_record_is_ignored(tmp_path: Path) -> None:
    records = [*branched()[:-1], last_prompt("no-such-uuid")]
    assert ingest(tmp_path, records).sendero.context_tokens == 1102


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
    assert ingest(tmp_path, compacted()).sendero.dropped_tokens == 1_000_823 - 22_191


def test_compact_summary_is_indexed_as_its_own_role(tmp_path: Path) -> None:
    nodes = {n.uuid: n for n in ingest(tmp_path, compacted()).nodes}
    assert nodes["s1"].role == "summary"
    assert "worktrees" in nodes["s1"].text


def test_pre_compaction_turns_survive_the_new_root(tmp_path: Path) -> None:
    """The boundary has parentUuid null, so the live walk cannot reach behind it."""
    delta = ingest(tmp_path, compacted())
    assert "a long piece of work" in [n.text for n in delta.nodes if n.text]
    assert delta.sendero.n_user_turns == 2


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
    assert delta.sendero.dropped_tokens == 1_000_823 - 22_191


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
    assert delta.sendero.context_tokens == 2 + 999_154 + 628


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
    assert delta.sendero.output_tokens == 1000


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
    assert delta.sendero.output_tokens == 20


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
    assert ingest(tmp_path, records).sendero.context_tokens == 1102


# --- titles, cwd, forks ----------------------------------------------------


def test_custom_title_beats_ai_title(tmp_path: Path) -> None:
    records = [*linear(), ai_title("generated"), custom_title("mine")]
    assert ingest(tmp_path, records).sendero.title == "mine"


def test_ai_title_beats_the_first_prompt(tmp_path: Path) -> None:
    assert ingest(tmp_path, [*linear(), ai_title("generated")]).sendero.title == "generated"


def test_slug_is_used_when_there_is_no_title(tmp_path: Path) -> None:
    records = linear()
    records[3]["slug"] = "pure-treehouse"
    assert ingest(tmp_path, records).sendero.title == "pure-treehouse"


def test_first_prompt_is_the_last_resort_title(tmp_path: Path) -> None:
    assert ingest(tmp_path, linear()).sendero.title == "why is conform relocating worktrees"


def test_relocation_moves_the_cwd(tmp_path: Path) -> None:
    records = [*linear(), relocated("/Users/dan/src/tide")]
    assert ingest(tmp_path, records).sendero.cwd == "/Users/dan/src/tide"


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


# --- discovery -------------------------------------------------------------


def test_discover_finds_sessions_and_subagents(tmp_path: Path) -> None:
    path = write(tmp_path, linear())
    subagents = path.parent / SESSION / "subagents"
    subagents.mkdir(parents=True)
    (subagents / "agent-abc123.jsonl").write_bytes(
        orjson.dumps(user("s1", None, "explore this", agentId="abc123", isSidechain=True)) + b"\n"
    )
    (subagents / "agent-abc123.meta.json").write_bytes(
        orjson.dumps({"agentType": "Explore", "description": "look around", "spawnDepth": 1})
    )

    found = {d.id for d in ClaudeSource(tmp_path).discover()}
    assert found == {f"claude:{SESSION}", f"claude:{SESSION}/abc123"}


def test_subagent_is_a_sendero_spawned_by_its_session(tmp_path: Path) -> None:
    write(tmp_path, linear())
    subagents = tmp_path / "-Users-dan-src-wormhole" / SESSION / "subagents"
    subagents.mkdir(parents=True)
    (subagents / "agent-abc123.jsonl").write_bytes(
        orjson.dumps(user("s1", None, "explore this", agentId="abc123", isSidechain=True)) + b"\n"
    )
    (subagents / "agent-abc123.meta.json").write_bytes(orjson.dumps({"agentType": "Explore"}))

    source = ClaudeSource(tmp_path)
    delta = source.ingest(subagents / "agent-abc123.jsonl")
    assert delta is not None
    assert delta.sendero.id == f"claude:{SESSION}/abc123"
    assert delta.sendero.is_sidechain
    assert delta.sendero.agent_type == "Explore"
    (edge,) = delta.edges
    assert edge == type(edge)(
        child=f"claude:{SESSION}/abc123", parent=f"claude:{SESSION}", kind="spawn", at_uuid=None
    )


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
    assert delta.sendero.n_user_turns == 2


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
    assert delta.sendero.n_user_turns == 1


def test_an_interruption_i_actually_quoted_is_kept(tmp_path: Path) -> None:
    """Only the bare sentinel is synthetic; the same words inside a real prompt are mine."""
    delta = ingest(
        tmp_path,
        [user("u1", None, "why does [Request interrupted by user] keep appearing?")],
    )
    assert delta.sendero.n_user_turns == 1


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
    assert delta.sendero.n_user_turns == 2
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
