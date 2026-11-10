"""The window's look and plain words: the tables' chances, the theme chosen, Coverage's words, and
a forecast Vesta made after kick-off (log schema 12) marked with Vesta's own sentence."""

import shutil
from pathlib import Path

from sample_log import COMMIT, VERSION, result
from small_world import TO_COME

from desk import theme
from desk.log import ForecastLog
from desk.opening import CheckedFile, Coverage
from desk.table import read_percent
from desk.words import describe_digest, describe_file, describe_forecast, describe_reason
from vesta.dataset_proof import Proof
from vesta.forecast_log.store import open_log
from vesta.forecast_log.writing import CAVEATS, Caveat, Publication, PublishedDistribution, publish


def test_a_chance_as_shown_is_read_as_a_share_of_its_bar():
    assert read_percent("43%") == 0.43
    assert read_percent("<1%") == 0.005
    assert read_percent(">99%") == 0.995
    assert read_percent("–") is None
    assert read_percent("") is None


def test_the_person_s_look_comes_before_the_system_s(monkeypatch):
    monkeypatch.setattr(theme, "read_system_theme", lambda: theme.DARK)
    assert theme.choose_theme(None) == theme.DARK
    assert theme.choose_theme(theme.LIGHT) == theme.LIGHT
    monkeypatch.setattr(theme, "read_system_theme", lambda: None)
    assert theme.choose_theme("purple") == theme.LIGHT, "light when nothing can be read"


def coverage(code, name, served, model=None, reason="", validation=None, missing=None):
    return Coverage(code, name, "italy/serie-a", served, model, reason, validation, missing or [])


def test_coverage_says_in_plain_words_what_vesta_does():
    from vesta.capabilities import FAILED, NATIONS_MODEL, _explain_basic_model

    present = {"B3": True, "B5": True}
    basic = coverage("SA", "Serie A", True, "B0", _explain_basic_model(True, present),
                     "not validated on this dataset")  # fmt: skip
    assert describe_forecast(basic) == "Yes · basic model · untested here"
    said = describe_reason(basic)
    assert said.startswith("The basic model, as the dataset's shots carry no expected goals")
    assert not any(word in said for word in ("B0", "B3", "M1", "M4", "lineups"))
    nations = coverage("UNL", "UEFA Nations League", True, "nations Elo", NATIONS_MODEL, FAILED)
    assert describe_forecast(nations) == "Yes · nations model · failed Vesta's tests"
    assert describe_reason(nations).startswith("The nations model, which did not pass")
    cup = coverage("CL", "UEFA Champions League", False, missing=["Premier League", "La Liga"])
    assert describe_reason(cup) == "Needs the files of Premier League, La Liga too."
    unknown = coverage("XL", "X League", False, reason="something Vesta says")
    assert describe_reason(unknown) == "Something Vesta says", "an unknown reason as Vesta wrote it"
    whole = CheckedFile(Path("11footy.db"), "all", frozenset({"SA"}), "0", False,
                        Proof(True, "listed", ""), True)  # fmt: skip
    assert describe_file(whole, [basic]) == "Everything: every competition and the players"
    assert describe_digest(whole, checked=False) == "Checked on first opening"
    league = CheckedFile(Path("italy.serie-a.db"), "italy/serie-a", frozenset({"SA"}), "0", True,
                         Proof(True, "listed", ""))  # fmt: skip
    assert describe_file(league, [basic]) == "Serie A"
    assert describe_digest(league, checked=True) == "Matches SHA256SUMS"


def test_a_forecast_made_after_kickoff_is_marked_with_vesta_s_sentence(sample_log, tmp_path):
    log = Path(shutil.copy(sample_log, tmp_path / "forecast-log.db"))
    connection = open_log(log)
    publish(
        connection,
        Publication(
            "pre_match",
            "after_kickoff",
            "2026-10-25T16:00:00Z",
            COMMIT,
            match_id=TO_COME,
            scheduled_kickoff_utc="2026-10-25T15:30:00Z",
        ),  # fmt: skip
        [PublishedDistribution("M1.result", TO_COME, VERSION, result(0.4, 0.3))],
        caveats=[Caveat("results", "forecast_after_kickoff", VERSION)],
    )
    connection.close()
    found = ForecastLog(log).read_result_forecast(TO_COME)
    assert found["after_kickoff"] and found["home"] == 0.4
    assert found["after_kickoff_caveat"] == CAVEATS["forecast_after_kickoff"]
    before = ForecastLog(sample_log).read_result_forecast(TO_COME)
    assert before is None or not before["after_kickoff"]
