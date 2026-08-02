"""Claude Code transcripts.

One `~/.claude/projects/<encoded-cwd>/<session-uuid>.jsonl` per sendero, plus
`<session-uuid>/subagents/agent-*.jsonl` for the agents it spawned. The records
form a DAG via parentUuid, not a list: rewinding an edited message branches it,
and compaction starts a fresh root inside the same file. So the live thread is
the walk back from the leaf, never the order the lines happen to be in.
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import orjson

from senderos.models import Compaction, Delta, Discovered, Edge, Node, Sendero

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
        found = []
        for path in sorted(self.root.glob("*/*.jsonl")):
            found.append(_discovered(f"claude:{path.stem}", path))
            for agent in sorted((path.parent / path.stem / "subagents").glob("agent-*.jsonl")):
                agent_id = agent.stem.removeprefix("agent-")
                found.append(_discovered(f"claude:{path.stem}/{agent_id}", agent, agent_id))
        return found

    def ingest(self, path: Path) -> Delta | None:
        return parse(path, _read(path))


def _discovered(id: str, path: Path, agent_id: str | None = None) -> Discovered:
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

    sendero = _sendero(path, records, nodes)
    branch_points = _branch_points(nodes)
    active = _active_branch(nodes, sendero.leaf_uuid)

    dag = _nodes(sendero.id, nodes, branch_points)
    compactions = _compactions(sendero.id, records)
    _measure(sendero, nodes, active, dag, compactions)

    return Delta(
        sendero=sendero, nodes=dag, edges=_edges(sendero.id, records), compactions=compactions
    )


def _sendero(path: Path, records: list[dict[str, Any]], nodes: list[dict[str, Any]]) -> Sendero:
    first, last = nodes[0], nodes[-1]
    session_id = first.get("sessionId") or path.stem
    agent_id = first.get("agentId")
    native_id = f"{session_id}/{agent_id}" if agent_id else session_id

    state = _sidecar_state(records)
    return Sendero(
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
        agent_type=_agent_type(path) if agent_id else None,
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


def _agent_type(path: Path) -> str | None:
    meta = path.with_suffix(".meta.json")
    if not meta.exists():
        return None
    return orjson.loads(meta.read_bytes()).get("agentType")


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
    node = by_uuid.get(leaf_uuid or "") or nodes[-1]
    chain = []
    seen = set()
    while node and node["uuid"] not in seen:
        seen.add(node["uuid"])
        chain.append(node)
        node = by_uuid.get(node.get("parentUuid") or "")
    chain.reverse()
    return chain


def _nodes(sendero_id: str, nodes: list[dict[str, Any]], branch_points: set[str]) -> list[Node]:
    """The whole DAG, prose or not, on every branch — abandoned ones included.

    A thread is frequently rewound at a tool call or a `turn_duration` record
    rather than at something anyone said, so keeping only prose would hide most
    of the branching: 9 branch points survive that filter, against 304 real ones.
    """
    return [
        Node(
            sendero_id=sendero_id,
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


def _compactions(sendero_id: str, records: list[dict[str, Any]]) -> list[Compaction]:
    out = []
    for record in records:
        if record.get("subtype") != "compact_boundary":
            continue
        meta = record.get("compactMetadata", {})
        preserved = meta.get("preservedMessages", {})
        out.append(
            Compaction(
                sendero_id=sendero_id,
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


def _edges(sendero_id: str, records: list[dict[str, Any]]) -> list[Edge]:
    """A fork reuses the parent's message uuid as its own root, so the splice is exact.

    A subagent's id embeds the session that spawned it.
    """
    for record in records:
        if fork := record.get("forkedFrom"):
            return [
                Edge(
                    child=sendero_id,
                    parent=f"claude:{fork['sessionId']}",
                    kind="fork",
                    at_uuid=fork.get("messageUuid"),
                )
            ]
    native = sendero_id.removeprefix("claude:")
    if "/" in native:
        return [Edge(child=sendero_id, parent=f"claude:{native.split('/')[0]}", kind="spawn")]
    return []


def _measure(
    sendero: Sendero,
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

    sendero.n_messages = len(messages)
    sendero.n_user_turns = sum(1 for n in dag if n.role == "user" and n.text)
    sendero.started_at = min(stamps, default=None)
    sendero.ended_at = max(stamps, default=None)
    sendero.model = next(
        (m for n in reversed(assistants) if (m := n.get("message", {}).get("model"))), None
    )
    sendero.context_tokens = next((c for n in reversed(assistants) if (c := _context_tokens(n))), 0)
    sendero.output_tokens = _output_tokens(assistants)
    sendero.dropped_tokens = max((c.pre_tokens - c.post_tokens for c in compactions), default=0)
    if not sendero.title:
        sendero.title = _title_from(dag)


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
