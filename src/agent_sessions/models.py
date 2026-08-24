"""The shape of a session, independent of which agent produced it."""

from collections.abc import Sequence
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


@dataclass(frozen=True, slots=True)
class Said:
    """Prose, by one side or the other.

    `uuid` is the record it came from, which is how a rendering of a turn is
    matched back to the turn: position drifts and text is not unique.
    """

    role: str
    text: str
    uuid: str = ""


@dataclass(frozen=True, slots=True)
class Ran:
    """A tool call and what came back from it, kept together.

    They arrive as separate records — the call on one, the result on the next —
    and every reader wants them as one thing.
    """

    tool: str
    input: dict[str, object]
    output: str = ""
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class Boundary:
    """Where a compaction cut the thread."""

    trigger: str
    pre_tokens: int
    post_tokens: int


# A conversation, in the order it happened, in the only shape a reader needs.
# One parser produces these; each place that shows a conversation is a
# formatter over them and nothing more.
Block = Said | Ran | Boundary


@dataclass(frozen=True, slots=True)
class Turn:
    """What I asked, and everything that followed until I asked again."""

    key: str
    asked: str
    blocks: list[Block]

    @property
    def whole(self) -> list[Block]:
        """The turn as it happened. A thread shows the answer; a reader wants both."""
        return [Said(role="user", text=self.asked, uuid=self.key), *self.blocks]


def turns(blocks: Sequence[Block]) -> list[Turn]:
    """The conversation as turns, which is the unit both readers ask for.

    Work before the first thing I said is not a turn: a window can open in the
    middle of a session, and the tail of an earlier exchange is not mine to
    render as an answer.
    """
    found: list[Turn] = []
    for block in blocks:
        if isinstance(block, Said) and block.role == "user":
            found.append(Turn(key=block.uuid, asked=block.text, blocks=[]))
        elif found:
            found[-1].blocks.append(block)
    return found


def turn_at(blocks: Sequence[Block], uuid: str) -> Turn | None:
    """The turn a point falls in, whether the point is what was asked or an answer.

    `show --turns` prints prompts and `tree` prints replies, so a point handed
    back can be either, and both name the same exchange.
    """
    return next(
        (
            turn
            for turn in turns(blocks)
            if uuid in {turn.key, *(b.uuid for b in turn.blocks if isinstance(b, Said))}
        ),
        None,
    )


@dataclass(slots=True)
class Delta:
    """Everything one transcript file contributed to the index."""

    session: Session
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    compactions: list[Compaction] = field(default_factory=list)
