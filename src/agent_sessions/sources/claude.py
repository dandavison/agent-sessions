"""Claude Code transcripts.

One `~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl` per session. The
records form a DAG via parentUuid, not a list: rewinding an edited message
branches it, and compaction starts a fresh root inside the same file. So the
live thread is the walk back from the leaf, never the order the lines happen
to be in.

Subagent transcripts, under `<session-uuid>/subagents/`, are not sessions. The
point of the index is work I took part in, and nobody talked to those.
"""

import os
from collections import defaultdict
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson

from agent_sessions.models import Compaction, Delta, Discovered, Edge, Node, Running, Session

PROJECTS_DIR = Path.home() / ".claude" / "projects"

# Records that are nodes in the DAG. Everything else is either sidecar state or
# noise: `attachment` carries injected context, `system` mostly turn timings.
NODE_TYPES = frozenset({"user", "assistant", "system", "attachment"})

TITLE_LIMIT = 120

# Synthetic user records written when a turn is interrupted. `interruptedMessageId`
# marks most of them, but 9% of the 187 in the corpus carry no flag at all, so the
# text is the only reliable signal. Both strings are fixed.
INTERRUPTIONS = frozenset(
    {"[Request interrupted by user]", "[Request interrupted by user for tool use]"}
)


class ClaudeSource:
    name = "claude"

    def __init__(self, root: Path = PROJECTS_DIR) -> None:
        self.root = root

    def discover(self) -> list[Discovered]:
        """Only top-level sessions: the glob does not descend into `subagents/`."""
        return [_discovered(f"claude:{p.stem}", p) for p in sorted(self.root.glob("*/*.jsonl"))]

    def ingest(self, path: Path) -> Delta | None:
        return parse(path, _read(path))

    def live(self) -> dict[str, Running]:
        return live()

    def render(self, path: Path, tools: bool, whole: bool) -> Iterator[str]:
        return render(_read(path), tools=tools, whole=whole)

    def resume_command(self, native_id: str, fork: bool = False) -> str:
        return f"claude -r {native_id}" + (" --fork-session" if fork else "")

    def resumable_from(self, path: Path, cwd: str) -> bool:
        return path.parent.name == encode(cwd)

    def fork_at(self, path: Path, at_uuid: str) -> str:
        return fork_at(path, at_uuid)


def encode(cwd: str) -> str:
    """Claude's name for a cwd's transcript directory. Holds for all 506 of mine."""
    return cwd.replace("/", "-").replace(".", "-")


def _discovered(id: str, path: Path) -> Discovered:
    stat = path.stat()
    return Discovered(id=id, path=path, size=stat.st_size, mtime=int(stat.st_mtime))


def _read(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                records.append(orjson.loads(line))
    return records


def parse(path: Path, records: list[dict[str, Any]]) -> Delta | None:
    """Fold one transcript into the index's model of it.

    None for the sidecar-only files a session leaves when it is started and
    abandoned: a title and a mode, but nothing ever said.
    """
    nodes = [r for r in records if r.get("type") in NODE_TYPES and r.get("uuid")]
    if not nodes:
        return None

    session = _session(path, records, nodes)
    branch_points = _branch_points(nodes)
    active = _active_branch(nodes, session.leaf_uuid)

    dag = _nodes(session.id, nodes, branch_points)
    compactions = _compactions(session.id, records)
    _measure(session, nodes, active, dag, compactions)

    return Delta(
        session=session, nodes=dag, edges=_edges(session.id, records), compactions=compactions
    )


def _session(path: Path, records: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> Session:
    first, last = nodes[0], nodes[-1]
    native_id = first.get("sessionId") or path.stem
    state = _sidecar_state(records)
    return Session(
        id=f"claude:{native_id}",
        agent="claude",
        native_id=native_id,
        path=str(path),
        cwd=state.get("relocatedCwd") or first.get("cwd"),
        git_branch=first.get("gitBranch"),
        title=state.get("customTitle") or state.get("aiTitle") or last.get("slug"),
        leaf_uuid=state.get("leafUuid"),
        is_sidechain=bool(first.get("isSidechain")),
        session_kind=first.get("sessionKind"),
        file_mtime=int(path.stat().st_mtime),
    )


def _sidecar_state(records: list[dict[str, Any]]) -> dict[str, str]:
    """Sidecar records are last-write-wins key/value updates appended to the file."""
    keys = {
        "custom-title": "customTitle",
        "ai-title": "aiTitle",
        "last-prompt": "leafUuid",
        "relocated": "relocatedCwd",
    }
    state: dict[str, str] = {}
    for record in records:
        key = keys.get(record.get("type", ""))
        if key and record.get(key):
            state[key] = record[key]
    return state


def _branch_points(nodes: list[dict[str, Any]]) -> set[str]:
    """A node the thread was rewound to: more than one record claims it as parent."""
    children: dict[str, set[str]] = defaultdict(set)
    for node in nodes:
        if parent := node.get("parentUuid"):
            children[parent].add(node["uuid"])
    return {uuid for uuid, kids in children.items() if len(kids) > 1}


def _active_branch(nodes: list[dict[str, Any]], leaf_uuid: str | None) -> list[dict[str, Any]]:
    """The live thread: walk parents back from the leaf, then read it forwards.

    Without a `last-prompt` record — 1 file in 3 has none — the leaf is simply
    the last record written.
    """
    by_uuid = {n["uuid"]: n for n in nodes}
    return _thread_to(by_uuid, by_uuid.get(leaf_uuid or "") or nodes[-1])


def _thread_to(
    by_uuid: dict[str, dict[str, Any]], node: dict[str, Any] | None
) -> list[dict[str, Any]]:
    chain = []
    seen = set()
    while node and node["uuid"] not in seen:
        seen.add(node["uuid"])
        chain.append(node)
        node = by_uuid.get(node.get("parentUuid") or "")
    chain.reverse()
    return chain


def _nodes(session_id: str, nodes: list[dict[str, Any]], branch_points: set[str]) -> list[Node]:
    """The whole DAG, prose or not, on every branch — abandoned ones included.

    A thread is frequently rewound at a tool call or a `turn_duration` record
    rather than at something anyone said, so keeping only prose would hide most
    of the branching: 9 branch points survive that filter, against 304 real ones.
    """
    return [
        Node(
            session_id=session_id,
            uuid=node["uuid"],
            parent_uuid=node.get("parentUuid"),
            seq=seq,
            role=_role(node),
            ts=_epoch(node.get("timestamp")),
            text=_text(node),
            request_id=node.get("requestId"),
            context_tokens=_context_tokens(node),
            is_branch_point=node["uuid"] in branch_points,
        )
        for seq, node in enumerate(nodes)
    ]


def _role(node: dict[str, Any]) -> str:
    if node.get("isCompactSummary"):
        return "summary"
    return str(node.get("type"))


def _text(node: dict[str, Any]) -> str:
    """Prose only. Tool calls, results and interruptions are not things anyone typed."""
    if node.get("type") not in ("user", "assistant"):
        return ""
    if node.get("isMeta") or node.get("interruptedMessageId"):
        return ""
    content = node.get("message", {}).get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
        text = "\n".join(parts).strip()
    else:
        return ""
    return "" if text in INTERRUPTIONS else text


def _compactions(session_id: str, records: list[dict[str, Any]]) -> list[Compaction]:
    out = []
    for record in records:
        if record.get("subtype") != "compact_boundary":
            continue
        meta = record.get("compactMetadata", {})
        preserved = meta.get("preservedMessages", {})
        out.append(
            Compaction(
                session_id=session_id,
                uuid=record["uuid"],
                ts=_epoch(record.get("timestamp")),
                trigger=meta.get("trigger", ""),
                pre_tokens=meta.get("preTokens", 0),
                post_tokens=meta.get("postTokens", 0),
                logical_parent_uuid=record.get("logicalParentUuid"),
                anchor_uuid=preserved.get("anchorUuid"),
                preserved_count=len(preserved.get("uuids", [])),
            )
        )
    return out


def _edges(session_id: str, records: list[dict[str, Any]]) -> list[Edge]:
    """A fork reuses the parent's message uuid as its own root, so the splice is exact."""
    for record in records:
        if fork := record.get("forkedFrom"):
            return [
                Edge(
                    child=session_id,
                    parent=f"claude:{fork['sessionId']}",
                    kind="fork",
                    at_uuid=fork.get("messageUuid"),
                )
            ]
    return []


def _measure(
    session: Session,
    nodes: list[dict[str, Any]],
    active: list[dict[str, Any]],
    dag: list[Node],
    compactions: list[Compaction],
) -> None:
    """Counts and token totals.

    Turn counts cover the whole transcript, because everything I typed I really
    typed, even on a branch since abandoned — and a file can hold several roots,
    so the live thread is often not all of it. Context is different: it is the
    latest value on the live thread, and not a sum.
    """
    messages = [n for n in nodes if n.get("type") in ("user", "assistant")]
    stamps = [t for n in nodes if (t := _epoch(n.get("timestamp")))]
    assistants = [n for n in active if n.get("type") == "assistant"]

    session.n_messages = len(messages)
    session.n_user_turns = sum(1 for n in dag if n.role == "user" and n.text)
    session.started_at = min(stamps, default=None)
    session.ended_at = max(stamps, default=None)
    session.model = next(
        (m for n in reversed(assistants) if (m := n.get("message", {}).get("model"))), None
    )
    session.context_tokens = next((c for n in reversed(assistants) if (c := _context_tokens(n))), 0)
    session.output_tokens = _output_tokens(assistants)
    session.dropped_tokens = max((c.pre_tokens - c.post_tokens for c in compactions), default=0)
    if not session.title:
        session.title = _title_from(dag)


def _output_tokens(assistants: list[dict[str, Any]]) -> int:
    """One API response spans several records sharing a requestId and one usage object."""
    seen: dict[str, int] = {}
    for i, node in enumerate(assistants):
        usage = node.get("message", {}).get("usage") or {}
        seen[node.get("requestId") or f"#{i}"] = usage.get("output_tokens", 0)
    return sum(seen.values())


def _context_tokens(node: dict[str, Any]) -> int:
    usage = node.get("message", {}).get("usage") or {}
    return (
        usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
    )


def _title_from(dag: list[Node]) -> str | None:
    """Last resort when the transcript carries no title of its own: my opening prompt."""
    first = next((n.text for n in dag if n.role == "user" and n.text), None)
    return " ".join(first.split())[:TITLE_LIMIT] if first else None


def _epoch(timestamp: str | None) -> int | None:
    if not timestamp:
        return None
    return int(datetime.fromisoformat(timestamp).timestamp())


def render(records: list[dict[str, Any]], tools: bool, whole: bool) -> Iterator[str]:
    """The transcript as markdown, read from the file rather than the index.

    The index holds no tool output by design, so this is the only way to see
    what was actually run. By default it follows the live thread; `whole`
    includes the branches that were abandoned.
    """
    nodes = [r for r in records if r.get("type") in NODE_TYPES and r.get("uuid")]
    if not whole:
        nodes = _active_branch(nodes, _sidecar_state(records).get("leafUuid"))
    boundaries = {c.uuid for c in _compactions("", records)}

    for node in nodes:
        if node["uuid"] in boundaries:
            yield from _compaction_rule(node)
        elif text := _text(node):
            yield f"\n## {_role(node)}\n\n{text}\n"
        elif tools:
            yield from _tool_calls(node)


def _compaction_rule(node: dict[str, Any]) -> Iterator[str]:
    meta = node.get("compactMetadata", {})
    yield f"\n---\n\n*Compacted ({meta.get('trigger')}): "
    yield f"{meta.get('preTokens', 0):,} → {meta.get('postTokens', 0):,} tokens*\n"


def _tool_calls(node: dict[str, Any]) -> Iterator[str]:
    content = node.get("message", {}).get("content")
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "tool_use":
            yield f"\n### {block.get('name')}\n\n```json\n{_json(block.get('input'))}\n```\n"
        elif block.get("type") == "tool_result":
            yield f"\n```\n{_result_text(block.get('content'))}\n```\n"


def _result_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return ""


def _json(value: Any) -> str:
    return orjson.dumps(value, option=orjson.OPT_INDENT_2).decode()


def fork_at(path: Path, at_uuid: str) -> str:
    """Write a session that ends where `at_uuid` did, and return its native id.

    Claude resumes a session at its leaf and nowhere else, so picking one up
    from earlier means handing it a session whose leaf is that point. This is
    what its own fork writes — the ancestry of the leaf, under a new session
    id, every record stamped with where it came from — stopped earlier. The
    session it came from is not touched.
    """
    records = _read(path)
    nodes = [r for r in records if r.get("type") in NODE_TYPES and r.get("uuid")]
    by_uuid = {n["uuid"]: n for n in nodes}
    if at_uuid not in by_uuid:
        raise ValueError(f"{at_uuid} is not a point in {path.name}.")

    thread = {n["uuid"] for n in _thread_to(by_uuid, by_uuid[at_uuid])}
    parent = nodes[0].get("sessionId") or path.stem
    native = str(uuid4())
    path.with_name(f"{native}.jsonl").write_bytes(
        b"".join(
            orjson.dumps(_forked(r, native, parent)) + b"\n"
            for r in records
            if r.get("uuid") in thread
        )
    )
    return native


def _forked(record: dict[str, Any], native: str, parent: str) -> dict[str, Any]:
    return record | {
        "sessionId": native,
        "forkedFrom": {"sessionId": parent, "messageUuid": record["uuid"]},
    }


SESSIONS_DIR = Path.home() / ".claude" / "sessions"


def live(sessions_dir: Path = SESSIONS_DIR) -> dict[str, Running]:
    """Session id -> what is running it, for the sessions running right now.

    Claude keeps a file per process here. They outlive the process that wrote
    them, so a pid that is gone means a stale entry, not a live session.
    """
    running = {}
    for path in sessions_dir.glob("*.json"):
        record = orjson.loads(path.read_bytes())
        pid = record.get("pid")
        if (sid := record.get("sessionId")) and _alive(pid):
            running[sid] = Running(pid=pid, status=str(record.get("status", "running")))
    return running


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError, ValueError):
        return False
    except PermissionError:
        return True
    return True
