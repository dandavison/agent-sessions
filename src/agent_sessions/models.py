"""The shape of a session, independent of which agent produced it."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Discovered:
    """A transcript file a source knows about, before it has been read."""

    id: str
    path: Path
    size: int
    mtime: int


@dataclass(slots=True)
class Session:
    id: str
    agent: str
    native_id: str
    path: str
    cwd: str | None = None
    project: str | None = None
    git_branch: str | None = None
    title: str | None = None
    model: str | None = None
    started_at: int | None = None
    ended_at: int | None = None
    n_user_turns: int = 0
    n_messages: int = 0
    context_tokens: int = 0
    output_tokens: int = 0
    dropped_tokens: int = 0
    leaf_uuid: str | None = None
    is_sidechain: bool = False
    session_kind: str | None = None
    file_mtime: int = 0


@dataclass(slots=True)
class Node:
    """One record in the transcript DAG.

    Every node is stored, not just the ones carrying prose: a thread is often
    rewound at a tool call or a timing record, so dropping those would lose the
    branch. `text` is empty for the silent ones, and only prose reaches search.
    """

    session_id: str
    uuid: str
    parent_uuid: str | None
    seq: int
    role: str
    ts: int | None
    text: str
    request_id: str | None = None
    context_tokens: int = 0
    is_branch_point: bool = False


@dataclass(frozen=True, slots=True)
class Edge:
    """A link between two sessions. Currently only `fork`."""

    child: str
    parent: str
    kind: str
    at_uuid: str | None = None


@dataclass(frozen=True, slots=True)
class Running:
    """A session with a live process behind it, and the pane it is sitting in."""

    pid: int
    status: str


@dataclass(frozen=True, slots=True)
class Compaction:
    session_id: str
    uuid: str
    ts: int | None
    trigger: str
    pre_tokens: int
    post_tokens: int
    logical_parent_uuid: str | None
    anchor_uuid: str | None
    preserved_count: int


@dataclass(slots=True)
class Delta:
    """Everything one transcript file contributed to the index."""

    session: Session
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
