"""A failure at start is written down and told, never swallowed."""

from desk import __main__ as start


def test_a_failure_at_start_is_logged_and_told(tmp_path, monkeypatch, capsys):
    shown = []
    monkeypatch.setattr(start, "show_in_a_box", lambda said: shown.append(said) or False)
    log = tmp_path / "11footy" / start.FAILURE_LOG
    try:
        raise RuntimeError("no display")
    except RuntimeError as error:
        said = start.report_start_failure(error, log)
    assert "RuntimeError: no display" in log.read_text(encoding="utf-8"), "the traceback is kept"
    assert said.startswith("11footy could not start: no display") and str(log) in said
    assert shown == [said]
    assert said in capsys.readouterr().err, "without Tk the words go to the error stream"


def test_a_failure_that_cannot_be_logged_carries_its_details(tmp_path, monkeypatch):
    monkeypatch.setattr(start, "show_in_a_box", lambda said: True)
    blocked = tmp_path / "a file"
    blocked.write_text("")
    try:
        raise ValueError("broken")
    except ValueError as error:
        said = start.report_start_failure(error, blocked / start.FAILURE_LOG)
    assert "Traceback" in said
