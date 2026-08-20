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
import socket
import sqlite3
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlsplit

import httpx

from agent_sessions import attend, channel, db, display, index, query, resume, topology
from agent_sessions.wormhole import WormholeUnavailable

HOST = "127.0.0.1"
EVERY_INTERFACE = "0.0.0.0"  # Asked for by name, never the default.
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
            return _resume(conn, id, remote=params.get("remote") in ("1", "true"))
        if id := _tail(path, "/issue/"):
            return _issue(conn, id)
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
    reverse = params.get("reverse") in ("1", "true")
    page = _page_number(params)
    try:
        # One more than fits, which is how the next page is known to exist
        # without counting the whole table.
        found = (
            query.search(conn, text, filters, LIMIT + 1, page * LIMIT)
            if text
            else query.recent(conn, filters, sort, LIMIT + 1, reverse, page * LIMIT)
        )
    except (ValueError, sqlite3.OperationalError) as e:
        return _page(
            "agent-sessions", _controls(params, sort) + _projects(conn), _banner(str(e)), status=400
        )

    more, found = len(found) > LIMIT, found[:LIMIT]
    live = _running()
    rows = "".join(_session_row(s, text, live) for s in found)
    body = (
        f"<table><thead><tr>{_headings(params, sort, reverse, sortable=not text)}"
        f"<th class=right></tr></thead><tbody>{rows}</tbody></table>"
        if found
        else "<p class=none>Nothing matched.</p>"
    )
    return _page(
        "agent-sessions",
        _controls(params, sort) + _projects(conn),
        _flash(params),
        body,
        _pager(params, page, len(found), more),
        _footer(conn),
    )


def _page_number(params: dict[str, str]) -> int:
    page = params.get("page", "0")
    return int(page) if page.isdigit() else 0


def _session_row(s: dict, text: str, live: set[str]) -> str:
    r = display.row(s)
    title = r["title"] or s["id"]
    note = (
        f"<div class=snippet>{_h(display.snippet(s['snippet'], text))}</div>"
        if text and s.get("snippet")
        else ""
    )
    project = _project(r["project"])
    dot = "<span class=live title='running now'>●</span>" if s["id"] in live else ""
    return (
        f"<tr><td class=when>{_h(r['when'])}<td>{project}<td class=num>{_h(r['context'])}"
        f"<td class=num>{r['turns']}"
        f"<td><a href='{_link(s['id'])}'>{_h(title)}</a>{dot}{note}"
        f"<td class=right>{_actions(s['id'])}</tr>"
    )


def _project(key: str) -> str:
    """A project, and which task of it, each filtering by itself.

    Wormhole's key names a task as `project:branch`, and most of the time it is
    the project that is wanted — everything done on that repo, whichever task it
    was done in — so that is what the first half links to.
    """
    if not key:
        return "<span class=none>—</span>"
    project, _, task = key.partition(":")
    cell = f"<a class=quiet href='/?{_h(urlencode({'project': project}))}'>{_h(project)}</a>"
    if task:
        cell += f"<a class=quiet href='/?{_h(urlencode({'project': key}))}'>:{_h(task)}</a>"
    return cell


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
        _controls(params, "recent") + _projects(conn),
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
        _controls(params, "recent") + _projects(conn),
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


def _resume(conn: sqlite3.Connection, id: str, remote: bool = False) -> Response:
    """A GET with an effect, deliberately: a link is the whole interface."""
    id, at = query.split_point(id)
    session = query.get(conn, id)
    if session is None:
        return _error(404, f"No single session matches {id!r}.")
    point = query.node(conn, session["id"], at) if at else None
    if at and point is None:
        return _error(404, f"No single point in {session['id']} matches {at!r}.")
    try:
        resumed = resume.resume(session, at=point["uuid"] if point else "", remote=remote)
    except resume.NotResumable as e:
        return _error(400, str(e))
    except (WormholeUnavailable, httpx.HTTPError) as e:
        return _error(502, str(e))

    # The conversation is not here any more, so neither is the browser that
    # asked for it. Leaving this page is the whole point of asking.
    if resumed.remote_home:
        return _redirect(resumed.remote_home)
    if resumed.at:
        message = (
            f"Resumed in {resumed.project} from {display.point(resumed.at)}, as a new session."
        )
    elif resumed.was_running and remote:
        message = (
            f"Already running ({resumed.was_running}), and remote control cannot be put on it"
            " from out here: type /rc in it, or set remoteControlAtStartup."
        )
    elif resumed.was_running:
        message = f"Already running ({resumed.was_running}); focused its pane in {resumed.project}."
    else:
        message = f"Resumed in {resumed.project}."
    return _redirect(f"{_link(session['id'])}?{urlencode({'msg': message})}")


def _issue(conn: sqlite3.Connection, id: str) -> Response:
    """Put the session on the control channel, and go to it. A link, again."""
    session = query.get(conn, id)
    if session is None:
        return _error(404, f"No single session matches {id!r}.")
    try:
        opened = attend.issue_for(control(), session)
    except (channel.NoToken, httpx.HTTPError) as e:
        return _error(502, str(e))
    return _redirect(opened.url)


def control() -> attend.Control:
    """The channel to reach, named here so a page can be tested without one."""
    return channel.Channel()


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
    """The search form, on every page, carrying whatever filters are in force.

    The order is chosen by clicking a column, so it travels as a hidden field
    rather than as a second control for the same thing.

    The project box names a list of what there is to filter by, which the pages
    that can read the index put beside it. It stays a text box: a name not on
    the list still works, and the error page has no list to offer.
    """
    order = f"<input type=hidden name=sort value='{_h(sort)}'>" + (
        "<input type=hidden name=reverse value=1>" if params.get("reverse") in ("1", "true") else ""
    )
    return (
        "<form class=controls action=/ method=get>"
        "<a class=brand href=/>agent-sessions</a>"
        f"<input name=q placeholder='what was said' value='{_h(params.get('q', ''))}'>"
        f"<input name=project placeholder=project list=projects"
        f" value='{_h(params.get('project', ''))}'>"
        f"<input name=since placeholder=since value='{_h(params.get('since', ''))}' size=6>"
        f"{order}"
        "<button type=submit>find</button>"
        "</form>"
    )


def _projects(conn: sqlite3.Connection) -> str:
    """Projects first, then the tasks of them: the coarse filter is the usual one."""
    projects, tasks = query.worked_in(conn)
    options = "".join(f"<option value='{_h(name)}'>" for name in (*projects, *tasks))
    return f"<datalist id=projects>{options}</datalist>"


# Column heading, what sorting by it is called, and the class its cells carry —
# which is how a narrow screen drops the columns it has no room for.
HEADINGS = (
    ("when", "recent", "when"),
    ("project", "project", ""),
    ("context", "context", "num"),
    ("turns", "turns", "num"),
    ("session", "title", ""),
)

# What a link out of this page carries with it. Not `page`: a new order or a new
# filter starts again from the top.
CARRIED = ("q", "project", "since", "agent", "min_context")


def _headings(params: dict[str, str], sort: str, reverse: bool, sortable: bool) -> str:
    """Clicking a column sorts by it, and clicking the one in force turns it around.

    A search has an order of its own — how well each session matched — so while
    one is in force the headings are only headings.
    """
    cells = []
    for label, name, css in HEADINGS:
        at = f"<th class='{css}'>"
        if not sortable:
            cells.append(f"{at}{label}")
            continue
        active = name == sort
        mark = f"<span class=arrow>{'▴' if reverse else '▾'}</span>" if active else ""
        wanted = {k: v for k, v in params.items() if k in CARRIED and v} | {"sort": name}
        if active and not reverse:
            wanted["reverse"] = "1"
        cells.append(f"{at}<a href='/?{_h(urlencode(wanted))}'>{label}</a>{mark}")
    return "".join(cells)


def _pager(params: dict[str, str], page: int, shown: int, more: bool) -> str:
    """Where in the list this is, and how to leave it. Fifty at a time."""
    if not page and not more:
        return ""
    first = page * LIMIT + 1
    return (
        f"<p class=pager>{_page_link(params, page - 1, 'newer') if page else ''}"
        f"<span class=quiet>{first}–{first + shown - 1}</span>"
        f"{_page_link(params, page + 1, 'older') if more else ''}</p>"
    )


def _page_link(params: dict[str, str], page: int, label: str) -> str:
    wanted = {k: v for k, v in params.items() if k in (*CARRIED, "sort", "reverse") and v}
    if page:
        wanted["page"] = str(page)
    return f"<a href='/?{_h(urlencode(wanted))}'>{label}</a>"


def _actions(id: str) -> str:
    """Where to pick the session up, out of the way until the row is being read.

    Named for the place the conversation lands, which is the only thing that
    differs between them. Not `local` and `remote`: there are two different
    remotes now — the agent's own remote control, and the control channel — and
    a word that means either means neither.
    """
    return (
        f"<a class=resume href='{_resume_link(id)}'>terminal</a>"
        f"<a class=resume href='/issue/{quote(id, safe=':')}'>issue</a>"
    )


def _link(id: str) -> str:
    return f"/session/{quote(id, safe=':')}"


def _resume_link(id: str, at: str = "", remote: bool = False) -> str:
    point = f"@{at}" if at else ""
    return f"/resume/{quote(id + point, safe=':@')}" + ("?remote=1" if remote else "")


def _flash(params: dict[str, str]) -> str:
    return f"<p class=flash>{_h(params['msg'])}</p>" if params.get("msg") else ""


def _banner(text: str) -> str:
    return f"<p class=banner>{_h(text)}</p>"


def _footer(conn: sqlite3.Connection) -> str:
    total = conn.execute("SELECT count(*) FROM session").fetchone()[0]
    synced = display.date(db.synced_at(conn), with_time=True) or "never"
    return f"<footer>{total} sessions · synced {_h(synced)} · <a href=/sync>sync</a></footer>"


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
th a { color: inherit; text-decoration: none }
th a:hover { color: var(--fg) }
.arrow { color: var(--dim); margin-left: 3px }
.pager { display: flex; gap: 14px; align-items: baseline; margin: 14px 0; font-size: 13px }
.resume { color: var(--accent); font-size: 13px; white-space: nowrap; margin-left: 10px }
/* An action per row is clutter until the row is the one being read. */
td .resume { opacity: 0; transition: opacity .08s }
tr:hover td .resume, td .resume:focus-visible { opacity: 1 }
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

/* A phone, which is where picking a session up somewhere else is asked for.
   Nothing hovers, so an action that waits to be hovered is an action that is
   never found. */
@media (hover: none) {
  td .resume { opacity: 1 }
}
/* And no room for the columns that are only worth a glance. */
@media (max-width: 640px) {
  main { padding: 0 12px 40px }
  .num { display: none }
  input[name=q] { flex: 1 0 100% }
  .resume { margin-left: 14px }
  td, th { padding: 11px 6px }
}
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


def reachable(port: int) -> list[str]:
    """The addresses something that is not this machine could ask for the index at.

    The mDNS name first, because the router hands out a new address whenever it
    feels like it and the name outlives that. The address itself second, for
    the phones whose resolver does not do mDNS — most Android ones.
    """
    names = [name for name in (_mdns_name(), _lan_address()) if name]
    return [f"http://{name}:{port}/" for name in names]


def _mdns_name() -> str:
    """`<host>.local`, which is what this machine answers to on the local network."""
    name = socket.gethostname()
    return name if name.endswith(".local") else ""


def _lan_address() -> str:
    """The address of the interface that reaches the rest of the network.

    Asking a UDP socket where it would send names the right interface without
    sending anything or caring which of the several this machine has is the one
    a phone can get to. The address is documentation-only, so nothing is routed.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))
        return probe.getsockname()[0]
