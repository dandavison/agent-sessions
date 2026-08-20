"""Getting a phone to the index, which is the part a phone cannot be told.

A LAN address is assigned by the router and changes; nobody wants to read one
off a laptop and type it into a phone. So `serve --lan` works out the addresses
itself and puts them on the terminal as a QR code, which is the one way of
handing a URL to a phone that costs nothing to do again when the address moves.
"""

from types import SimpleNamespace

import pytest

from agent_sessions import cli, web


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """What the socket was asked to bind, without binding it."""
    bound: list[tuple[str, int]] = []
    monkeypatch.setattr(web, "serve", lambda host, port: bound.append((host, port)))
    return bound


def test_by_default_nothing_outside_this_machine_can_reach_it(served, capsys) -> None:
    """The index is every conversation I have had. It stays here unless I say otherwise."""
    cli.run(["serve", "--no-open"])
    assert served == [("127.0.0.1", web.PORT)]


def test_asking_for_the_lan_binds_every_interface(served, capsys) -> None:
    cli.run(["serve", "--lan", "--no-open"])
    assert served == [("0.0.0.0", web.PORT)]


def test_a_lan_serve_names_an_address_a_phone_could_use(served, capsys) -> None:
    """Loopback is no use to a phone, and it is the only address `serve` knew about.

    It still says loopback too, which is where the browser on this machine goes.
    """
    cli.run(["serve", "--lan", "--no-open"])
    err = capsys.readouterr().err
    assert [url for url in web.reachable(web.PORT) if url in err]


def test_a_lan_serve_puts_the_address_on_the_terminal_as_a_qr_code(served, capsys) -> None:
    """Pointing a camera at it beats reading an address out and typing it in."""
    cli.run(["serve", "--lan", "--no-open"])
    assert "█" in capsys.readouterr().err


def test_a_local_serve_has_nothing_to_scan(served, capsys) -> None:
    cli.run(["serve", "--no-open"])
    assert "█" not in capsys.readouterr().err


def test_the_addresses_offered_are_ones_something_else_could_reach() -> None:
    urls = web.reachable(7118)
    assert urls
    assert all(url.startswith("http://") and url.endswith(":7118/") for url in urls)
    assert not [url for url in urls if "127.0.0.1" in url or "localhost" in url]


# --- and attending the control channel while it does ------------------------


def test_serving_does_not_attend_unless_asked(served, capsys, monkeypatch) -> None:
    """Serving is reading. Answering prompts runs an agent, which is not the same."""
    attended: list[object] = []
    monkeypatch.setattr(cli.attend, "loop", lambda *a, **k: attended.append(a))
    cli.run(["serve", "--no-open"])
    assert attended == []


def test_asking_it_to_attend_attends_beside_the_server(served, capsys, monkeypatch) -> None:
    """One process to leave running, rather than two terminals to remember."""
    attended: list[object] = []
    monkeypatch.setattr(cli.attend, "loop", lambda *a, **k: attended.append(a))
    monkeypatch.setattr(cli.channel, "Channel", lambda: SimpleNamespace(repo="dan/agent-work"))
    cli.run(["serve", "--attend", "--no-open"])
    assert attended
