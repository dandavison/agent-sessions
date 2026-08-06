"""A web UI over the index, in which a link is enough to resume.

Resuming is a GET, so the URL is clickable wherever a URL can be clicked — a row
in this page, a chat message, a note, an agent's output — and not only from a
form in this UI. That is the point of the thing: `/resume/claude:7e90a7c6` hands
the session to wormhole, which puts it back in the worktree it belongs to.

Requests are handled by `handle`, which takes a path and a query string and
returns a `Response`. The socket is the adapter, so the pages are testable
without one.
"""

import html
import sqlite3
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import httpx

from agent_sessions import db, display, index, query, resume, topology
from agent_sessions.wormhole import WormholeUnavailable

HOST = "127.0.0.1"
PORT = 7118
LIMIT = 50


@dataclass(frozen=True, slots=True)
class Response:
    body: str
    status: int = 200
    content_type: str = "text/html; charset=utf-8"
    location: str = ""


def handle(path: str, query_string: str = "") -> Response:
    """Route one request. The only entry point; the server below is plumbing."""
    params = {k: v[0] for k, v in parse_qs(query_string).items()}
    conn = db.connect()
    try:
        if path == "/":
            return _sessions(conn, params)
        if path == "/sync":
            return _sync(conn)
        if id := _tail(path, "/resume/"):
            return _resume(conn, id, fork=params.get("fork") in ("1", "true"))
        if id := _tail(path, "/transcript/"):
            return _transcript(conn, id, params)
        if id := _tail(path, "/session/"):
            return _session(conn, id, params)
        return _error(404, "No such page.")
    finally:
        conn.close()


def _tail(path: str, prefix: str) -> str:
    return path[len(prefix) :] if path.startswith(prefix) else ""


# --- pages -----------------------------------------------------------------


def _sessions(conn: sqlite3.Connection, params: dict[str, str]) -> Response:
    text = params.get("q", "").strip()
    filters = query.Filters(
        project=params.get("project") or None,
        agent=params.get("agent") or None,
        since=params.get("since") or None,
        min_context=params.get("min_context") or None,
    )
    sort = params.get("sort", "") if params.get("sort") in query.SORTS else "recent"
    try:
        found = (
            query.search(conn, text, filters, LIMIT)
            if text
            else query.recent(conn, filters, sort, LIMIT)
        )
    except (ValueError, sqlite3.OperationalError) as e:
        return _page("agent-sessions", _controls(params, sort), _banner(str(e)), status=400)

    live = _running()
    rows = "".join(_session_row(s, text, live) for s in found)
    body = (
        f"<table><thead><tr><th>when<th>project<th>context<th>turns"
        f"<th>session<th class=right>resume</tr></thead><tbody>{rows}</tbody></table>"
        if found
        else "<p class=none>Nothing matched.</p>"
    )
    return _page(
        "agent-sessions",
        _controls(params, sort),
        _flash(params),
        body,
        _footer(conn, len(found)),
    )


def _session_row(s: dict, text: str, live: set[str]) -> str:
    r = display.row(s)
    title = r["title"] or s["id"]
    note = (
        f"<div class=snippet>{_h(display.snippet(s['snippet'], text))}</div>"
        if text and s.get("snippet")
        else ""
    )
    project = (
        f"<a class=quiet href='/?{urlencode({'project': r['project']})}'>{_h(r['project'])}</a>"
        if r["project"]
        else "<span class=none>—</span>"
    )
    dot = "<span class=live title='running now'>●</span>" if s["id"] in live else ""
    return (
        f"<tr><td class=when>{_h(r['when'])}<td>{project}<td class=num>{_h(r['context'])}"
        f"<td class=num>{r['turns']}"
        f"<td><a href='{_link(s['id'])}'>{_h(title)}</a>{dot}{note}"
        f"<td class=right>{_actions(s['id'])}</tr>"
    )


def _session(conn: sqlite3.Connection, id: str, params: dict[str, str]) -> Response:
    session = query.get(conn, id)
    if session is None:
        return _error(404, f"No single session matches {id!r}.")

    shape = topology.of(conn, session)
    fields = "".join(
        f"<div class=field><dt>{_h(k)}<dd>{_h(str(v))}</div>"
        for k, v in display.details(session).items()
        if v
    )
    said = "".join(_turn(session["id"], t) for t in query.turns(conn, session["id"]))
    return _page(
        session["title"] or session["id"],
        _controls(params, "recent"),
        _flash(params),
        f"<h1>{_h(session['title'] or session['id'])}</h1>",
        f"<p class=actions>{_actions(session['id'])}"
        f" <a class=button href='/transcript/{quote(session['id'], safe=':')}'>transcript</a></p>",
        f"<dl class=details>{fields}</dl>",
        f"<h2>shape</h2>{_tree(session['id'], shape)}",
        f"<h2>turns</h2><ol class=turns>{said}</ol>" if said else "",
    )


def _turn(id: str, t: dict) -> str:
    d = display.turn(t)
    context = f"<span class=num>{_h(d['context'])}</span>" if d["context"] else ""
    point = f"<a class=point href='{_resume_link(id, at=t['uuid'])}'>{_h(d['at'])}</a>"
    return (
        f"<li class=turn><div class=meta>{_h(d['when'])} · {_h(d['role'])} {context} {point}</div>"
        f"<div class=text>{_h(d['text'])}</div>"
    )


def _tree(id: str, shape: topology.Topology) -> str:
    parts = []
    if origin := shape.forked_from:
        parts.append(
            f"<p class=edge>forked from <a href='{_link(origin['parent'])}'>"
            f"{_h(origin['parent'])}</a> at {_h((origin['at_uuid'] or '')[:8])}</p>"
        )
    parts.append(f"<ul class=tree>{''.join(_branch(id, root) for root in shape.roots)}</ul>")
    for fork in shape.forks:
        parts.append(
            f"<p class=edge>fork → <a href='{_link(fork['child'])}'>{_h(fork['child'])}</a>"
            f" at {_h((fork['at_uuid'] or '')[:8])}</p>"
        )
    return "".join(parts)


def _branch(id: str, segment: topology.Segment) -> str:
    """The label is styled, not the item: strikethrough on a list item reaches its children.

    Each stretch links to picking the session up where that stretch ended, which
    is the reason to be reading its shape at all.
    """
    classes = " ".join(
        c
        for c, on in (("compaction", bool(segment.compaction)), ("abandoned", segment.abandoned))
        if on
    )
    below = (
        f"<ul>{''.join(_branch(id, child) for child in segment.children)}</ul>"
        if segment.children
        else ""
    )
    return (
        f"<li><span class='{classes}'>{_h(display.describe(segment))}</span> "
        f"<a class=point href='{_resume_link(id, at=segment.end)}'>"
        f"{_h(display.point(segment.end))}</a>{below}"
    )


def _transcript(conn: sqlite3.Connection, id: str, params: dict[str, str]) -> Response:
    session = query.get(conn, id)
    if session is None:
        return _error(404, f"No single session matches {id!r}.")
    path = Path(session["path"])
    if not path.exists():
        return _error(404, f"{path} is gone. Sync the index.")

    tools = params.get("tools") in ("1", "true")
    whole = params.get("whole") in ("1", "true")
    source = index.SOURCES[session["agent"]]
    markdown = "".join(source.render(path, tools=tools, whole=whole))
    return _page(
        session["title"] or session["id"],
        _controls(params, "recent"),
        f"<h1><a href='{_link(session['id'])}'>{_h(session['title'] or session['id'])}</a></h1>",
        f"<p class=actions>{_toggle(session['id'], 'tools', tools, whole)}"
        f"{_toggle(session['id'], 'whole', whole, tools)}</p>",
        f"<pre class=transcript>{_h(markdown)}</pre>",
    )


def _toggle(id: str, name: str, on: bool, other: bool) -> str:
    other_name = "whole" if name == "tools" else "tools"
    wanted = {name: "0" if on else "1", other_name: "1" if other else "0"}
    return (
        f"<a class='button{' on' if on else ''}' "
        f"href='/transcript/{quote(id, safe=':')}?{urlencode(wanted)}'>{name}</a> "
    )


def _resume(conn: sqlite3.Connection, id: str, fork: bool) -> Response:
    """A GET with an effect, deliberately: a link is the whole interface."""
    id, at = query.split_point(id)
    session = query.get(conn, id)
    if session is None:
        return _error(404, f"No single session matches {id!r}.")
    point = query.node(conn, session["id"], at) if at else None
    if at and point is None:
        return _error(404, f"No single point in {session['id']} matches {at!r}.")
    try:
        resumed = resume.resume(session, fork=fork, at=point["uuid"] if point else "")
    except resume.NotResumable as e:
        return _error(400, str(e))
    except (WormholeUnavailable, httpx.HTTPError) as e:
        return _error(502, str(e))

    if resumed.at:
        message = (
            f"Resumed in {resumed.project} from {display.point(resumed.at)}, as a new session."
        )
    elif resumed.was_running and not fork:
        message = f"Already running ({resumed.was_running}); focused its pane in {resumed.project}."
    else:
        message = f"{'Forked' if fork else 'Resumed'} in {resumed.project}."
    return _redirect(f"{_link(session['id'])}?{urlencode({'msg': message})}")


def _sync(conn: sqlite3.Connection) -> Response:
    result = index.sync(conn)
    return _redirect(
        "/?" + urlencode({"msg": f"Indexed {result.indexed} sessions, {result.nodes} nodes."})
    )


# --- markup ----------------------------------------------------------------


def _page(title: str, *sections: str, status: int = 200) -> Response:
    body = "".join(s for s in sections if s)
    return Response(
        f"<!doctype html><html lang=en><head><meta charset=utf-8>"
        f"<meta name=viewport content='width=device-width, initial-scale=1'>"
        f"<title>{_h(title)}</title><style>{CSS}</style></head>"
        f"<body><main>{body}</main></body></html>",
        status=status,
    )


def _controls(params: dict[str, str], sort: str) -> str:
    """The search form, on every page, carrying whatever filters are in force."""
    sorts = "".join(
        f"<option value={s}{' selected' if s == sort else ''}>{s}</option>"
        for s in sorted(query.SORTS)
    )
    return (
        "<form class=controls action=/ method=get>"
        "<a class=brand href=/>agent-sessions</a>"
        f"<input name=q placeholder='what was said' value='{_h(params.get('q', ''))}'>"
        f"<input name=project placeholder=project value='{_h(params.get('project', ''))}'>"
        f"<input name=since placeholder=since value='{_h(params.get('since', ''))}' size=6>"
        f"<select name=sort>{sorts}</select>"
        "<button type=submit>find</button>"
        "</form>"
    )


def _actions(id: str) -> str:
    return (
        f"<a class='button go' href='{_resume_link(id)}'>resume</a> "
        f"<a class=button href='{_resume_link(id, fork=True)}'>fork</a>"
    )


def _link(id: str) -> str:
    return f"/session/{quote(id, safe=':')}"


def _resume_link(id: str, fork: bool = False, at: str = "") -> str:
    point = f"@{at}" if at else ""
    return f"/resume/{quote(id + point, safe=':@')}" + ("?fork=1" if fork else "")


def _flash(params: dict[str, str]) -> str:
    return f"<p class=flash>{_h(params['msg'])}</p>" if params.get("msg") else ""


def _banner(text: str) -> str:
    return f"<p class=banner>{_h(text)}</p>"


def _footer(conn: sqlite3.Connection, shown: int) -> str:
    total = conn.execute("SELECT count(*) FROM session").fetchone()[0]
    synced = display.date(db.synced_at(conn), with_time=True) or "never"
    return (
        f"<footer>{shown} of {total} sessions · synced {_h(synced)} · "
        "<a href=/sync>sync</a></footer>"
    )


def _running() -> set[str]:
    return {
        f"{source.name}:{native_id}"
        for source in index.SOURCES.values()
        for native_id in source.live()
    }


def _error(status: int, message: str) -> Response:
    return _page("agent-sessions", _controls({}, "recent"), _banner(message), status=status)


def _redirect(location: str) -> Response:
    return Response("", status=303, location=location)


def _h(text: object) -> str:
    return html.escape("" if text is None else str(text))


CSS = """
:root { color-scheme: light dark; --fg: #1c1c1e; --bg: #fdfdfd; --dim: #6b6b70;
        --line: #e2e2e5; --accent: #0a58ca; --warn: #b3261e; --live: #1a7f37; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e8e8ea; --bg: #16171a; --dim: #9a9aa1; --line: #2c2e33;
          --accent: #79a9ff; --warn: #ff8a80; --live: #4ac26b; }
}
* { box-sizing: border-box }
body { margin: 0; background: var(--bg); color: var(--fg);
       font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif }
main { max-width: 1100px; margin: 0 auto; padding: 0 20px 60px }
a { color: var(--accent); text-decoration: none }
a:hover { text-decoration: underline }
h1 { font-size: 20px; margin: 24px 0 8px }
h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--dim);
     margin: 32px 0 8px; font-weight: 600 }
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
            padding: 14px 0; border-bottom: 1px solid var(--line); margin-bottom: 4px }
.brand { font-weight: 600; color: var(--fg); margin-right: 8px }
input, select, button { font: inherit; padding: 5px 8px; border: 1px solid var(--line);
                        border-radius: 6px; background: var(--bg); color: var(--fg) }
input[name=q] { flex: 1; min-width: 180px }
button { cursor: pointer }
table { width: 100%; border-collapse: collapse }
th { text-align: left; font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
     color: var(--dim); font-weight: 600; padding: 10px 8px; border-bottom: 1px solid var(--line) }
td { padding: 9px 8px; border-bottom: 1px solid var(--line); vertical-align: top }
tbody tr:hover { background: color-mix(in oklab, var(--fg) 4%, transparent) }
.when, .num { color: var(--dim); font-variant-numeric: tabular-nums; white-space: nowrap }
.num { text-align: right }
.right { text-align: right; white-space: nowrap }
.quiet { color: var(--dim) }
.none { color: var(--dim) }
.snippet { color: var(--dim); margin-top: 3px; font-size: 13px }
.live { color: var(--live); margin-left: 6px }
.button { display: inline-block; padding: 3px 9px; border: 1px solid var(--line);
          border-radius: 6px; font-size: 13px; white-space: nowrap }
.button.go { border-color: var(--accent) }
.button.on { background: color-mix(in oklab, var(--accent) 18%, transparent) }
.actions { display: flex; gap: 8px; margin: 12px 0 }
.flash { padding: 9px 12px; border-radius: 6px; margin: 14px 0;
         background: color-mix(in oklab, var(--live) 16%, transparent) }
.banner { padding: 9px 12px; border-radius: 6px; margin: 14px 0; color: var(--warn);
          background: color-mix(in oklab, var(--warn) 12%, transparent) }
.details { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
           gap: 10px 20px; margin: 12px 0 }
.field dt { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--dim) }
.field dd { margin: 1px 0 0; overflow-wrap: anywhere }
.tree, .tree ul { list-style: none; margin: 0; padding-left: 18px;
                  border-left: 1px solid var(--line) }
.tree { padding-left: 0; border: 0 }
.tree li { padding: 2px 0 }
.tree .compaction { color: var(--dim); font-style: italic }
.tree .abandoned { color: var(--dim); text-decoration: line-through }
.point { color: var(--dim); font-size: 12px; font-family: ui-monospace, SFMono-Regular, monospace }
.turns { list-style: none; padding: 0; margin: 0 }
.turn { border-top: 1px solid var(--line); padding: 12px 0 }
.turn .meta { color: var(--dim); font-size: 12px; margin-bottom: 4px }
.turn .text { white-space: pre-wrap; overflow-wrap: anywhere; max-height: 22em; overflow: auto }
.transcript { white-space: pre-wrap; overflow-wrap: anywhere; font-size: 13px;
              font-family: ui-monospace, SFMono-Regular, Menlo, monospace }
footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--line);
         color: var(--dim); font-size: 13px }
"""


# --- server ----------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = "agent-sessions"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 — the name is http.server's
        url = urlsplit(self.path)
        response = handle(url.path, url.query)
        payload = response.body.encode()
        self.send_response(response.status)
        if response.location:
            self.send_header("Location", response.location)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # http.server names it
        print(f"{self.command} {self.path}", file=sys.stderr)


def serve(host: str = HOST, port: int = PORT) -> None:
    with ThreadingHTTPServer((host, port), _Handler) as server:
        server.serve_forever()
