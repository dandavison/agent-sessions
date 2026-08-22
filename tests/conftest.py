"""Builders for Claude Code transcript records, shaped like the real thing."""

from typing import Any

import pytest

from agent_sessions import limits

SESSION = "7e90a7c6-ce43-4dfd-9d7c-8eb01ac7ccf2"


def user(uuid: str, parent: str | None, text: str, **kw: Any) -> dict[str, Any]:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": SESSION,
        "timestamp": _stamp(uuid),
        "cwd": "/Users/dan/src/wormhole",
        "gitBranch": "main",
        "isSidechain": False,
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        **kw,
    }


def assistant(
    uuid: str,
    parent: str | None,
    blocks: list[dict[str, Any]],
    request_id: str = "req_1",
    usage: dict[str, int] | None = None,
    **kw: Any,
) -> dict[str, Any]:
    return {
        "type": "assistant",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": SESSION,
        "requestId": request_id,
        "timestamp": _stamp(uuid),
        "cwd": "/Users/dan/src/wormhole",
        "gitBranch": "main",
        "isSidechain": False,
        "message": {
            "role": "assistant",
            "model": "claude-opus-5",
            "id": f"msg_{request_id}",
            "content": blocks,
            "usage": usage
            or {
                "input_tokens": 2,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 100,
                "output_tokens": 50,
            },
        },
        **kw,
    }


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_use(name: str = "Bash") -> dict[str, Any]:
    return {"type": "tool_use", "id": "toolu_1", "name": name, "input": {"command": "ls"}}


def tool_result(uuid: str, parent: str) -> dict[str, Any]:
    return {
        "type": "user",
        "uuid": uuid,
        "parentUuid": parent,
        "sessionId": SESSION,
        "timestamp": _stamp(uuid),
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "a\nb\n"}],
        },
    }


def last_prompt(leaf: str) -> dict[str, Any]:
    return {"type": "last-prompt", "lastPrompt": "…", "leafUuid": leaf, "sessionId": SESSION}


def ai_title(title: str) -> dict[str, Any]:
    return {"type": "ai-title", "aiTitle": title, "sessionId": SESSION}


def custom_title(title: str) -> dict[str, Any]:
    return {"type": "custom-title", "customTitle": title, "sessionId": SESSION}


def relocated(cwd: str) -> dict[str, Any]:
    return {"type": "relocated", "relocatedCwd": cwd, "sessionId": SESSION}


def compact_boundary(
    uuid: str,
    logical_parent: str,
    anchor: str,
    pre: int = 1_000_823,
    post: int = 22_191,
    trigger: str = "auto",
    preserved: int = 3,
) -> dict[str, Any]:
    return {
        "type": "system",
        "subtype": "compact_boundary",
        "content": "Conversation compacted",
        "uuid": uuid,
        "parentUuid": None,
        "logicalParentUuid": logical_parent,
        "sessionId": SESSION,
        "timestamp": _stamp(uuid),
        "compactMetadata": {
            "trigger": trigger,
            "preTokens": pre,
            "postTokens": post,
            "preservedSegment": {"anchorUuid": anchor},
            "preservedMessages": {
                "anchorUuid": anchor,
                "uuids": [f"p{i}" for i in range(preserved)],
            },
        },
    }


def compact_summary(uuid: str, parent: str, text: str) -> dict[str, Any]:
    return user(uuid, parent, text, isCompactSummary=True, isVisibleInTranscriptOnly=True)


def _stamp(uuid: str) -> str:
    """Distinct, ordered timestamps keyed off the uuid, so ordering is checkable."""
    n = sum(ord(c) for c in uuid) % 60
    return f"2026-07-18T01:{n:02d}:00.000Z"


@pytest.fixture(autouse=True)
def _own_ledger(tmp_path, monkeypatch):
    """No test may touch the real breaker.

    The first run of the limits tests tripped the live one and wrote fifty-nine
    records into the real ledger, which would have stopped the loop attending
    my actual issues. A safeguard that a test run can fire is not a safeguard.
    """
    monkeypatch.setattr(limits, "HOME", tmp_path)
    monkeypatch.setattr(limits, "BREAKER", tmp_path / "tripped")
    monkeypatch.setattr(limits, "LEDGER", tmp_path / "spending.jsonl")
