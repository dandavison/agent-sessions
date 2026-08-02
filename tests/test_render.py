import json

import pytest

from senderos.render import AGENT_ENV_VARS, MIN_WIDTH, Format, Renderer, _fit, detect

Rows = list[dict[str, object]]

LONG_TITLE = "a very long title indeed" * 4

ROWS: Rows = [
    {"id": "claude:aaaa", "project": "wormhole", "title": LONG_TITLE},
    {"id": "claude:bbbb", "project": "temporal:dan/x", "title": "short"},
]


def render(fmt: Format, capsys: pytest.CaptureFixture[str], rows: Rows = ROWS) -> str:
    Renderer(fmt).table(rows)
    return capsys.readouterr().out


def test_agent_format_is_tsv_and_untruncated(capsys: pytest.CaptureFixture[str]) -> None:
    out = render(Format.AGENT, capsys)
    header, first, second = out.splitlines()
    assert header.split("\t") == ["id", "project", "title"]
    assert first.split("\t")[2] == LONG_TITLE
    assert second.split("\t")[0] == "claude:bbbb"


def test_human_format_truncates(capsys: pytest.CaptureFixture[str]) -> None:
    out = render(Format.HUMAN, capsys)
    assert "…" in out
    assert LONG_TITLE not in out


def test_human_format_uppercases_headers(capsys: pytest.CaptureFixture[str]) -> None:
    assert render(Format.HUMAN, capsys).splitlines()[0].startswith("ID")


def test_json_format_round_trips(capsys: pytest.CaptureFixture[str]) -> None:
    assert json.loads(render(Format.JSON, capsys)) == ROWS


def test_quiet_format_is_ids_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert render(Format.QUIET, capsys).splitlines() == ["claude:aaaa", "claude:bbbb"]


def test_hints_go_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    Renderer(Format.AGENT).hint("run `senderos sync`")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "run `senderos sync`" in captured.err


def test_quiet_suppresses_hints(capsys: pytest.CaptureFixture[str]) -> None:
    Renderer(Format.QUIET).hint("nope")
    assert capsys.readouterr().err == ""


def test_cells_never_break_tsv(capsys: pytest.CaptureFixture[str]) -> None:
    rows: Rows = [{"id": "x", "text": "line one\nline two\twith tab"}]
    out = render(Format.AGENT, capsys, rows)
    assert len(out.splitlines()) == 2
    assert out.splitlines()[1].split("\t") == ["x", "line one line two with tab"]


def test_none_renders_empty(capsys: pytest.CaptureFixture[str]) -> None:
    out = render(Format.AGENT, capsys, [{"id": "x", "project": None}])
    assert out.splitlines()[1] == "x\t"


@pytest.mark.parametrize("var", AGENT_ENV_VARS)
def test_agent_env_var_selects_agent_format(var: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for v in AGENT_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(var, "1")
    assert detect() is Format.AGENT


def test_pipe_selects_agent_format(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in AGENT_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
    assert detect() is Format.AGENT


def test_tty_without_agent_env_selects_human(monkeypatch: pytest.MonkeyPatch) -> None:
    for v in AGENT_ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert detect() is Format.HUMAN


def test_empty_table_is_not_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    assert render(Format.AGENT, capsys, []) == ""
    assert render(Format.HUMAN, capsys, []).strip() == "(none)"


# --- column fitting --------------------------------------------------------


def test_narrow_columns_are_left_alone_when_one_is_long() -> None:
    """An id should not be squeezed as hard as a snippet."""
    assert _fit([10, 6, 400], 100) == [10, 6, 80]  # plus two 2-space gaps


def test_two_long_columns_share_what_is_left() -> None:
    assert _fit([10, 200, 200], 100) == [10, 43, 43]


def test_widths_that_already_fit_are_untouched() -> None:
    assert _fit([10, 20, 30], 200) == [10, 20, 30]


def test_a_very_narrow_terminal_still_produces_columns() -> None:
    assert all(w >= MIN_WIDTH for w in _fit([50, 50, 50], 10))


def test_long_values_are_marked_as_cut(capsys: pytest.CaptureFixture[str]) -> None:
    out = render(Format.HUMAN, capsys, [{"id": "x", "title": "y" * 500}])
    assert "…" in out
