"""The blind verdicts: the new models on the closed seasons (here the small world's 2016 and 2017
stand in for them), the reference of the result frequencies, and the rule that they run once."""

import tomllib
from pathlib import Path

from vesta.blind_verdicts import (
    find_earlier_report,
    judge_against_frequencies,
    make_blind_rules,
    run_blind,
)
from vesta.spine.match_forecast import MatchForecast

ROOT = Path(__file__).resolve().parents[3]
SPINE = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
RESEARCH = tomllib.loads((ROOT / "config/vesta/research.toml").read_text(encoding="utf-8"))
STAND_IN = {**RESEARCH, "seasons": {**RESEARCH["seasons"], "closed": [2016, 2017]}}


def test_the_new_models_run_on_the_closed_seasons_or_say_why(world, tmp_path):
    verdicts = run_blind(tmp_path / "football-world.db", SPINE, STAND_IN, say=lambda _: None)
    assert [verdict.name for verdict in verdicts] == [
        "The UEFA cups", "The national teams", "Goals in time (step 3)", "Ties (step 3b)",
    ]  # the spine is measured forward, in public
    for verdict in verdicts:
        assert verdict.passed is not None or verdict.note.startswith("could not run")
    assert make_blind_rules(RESEARCH).min_seasons_improved == 2  # both closed seasons


def _forecast(match_id, season, probabilities, result):
    return MatchForecast(match_id, "CL", season, f"{season}-10-01T19:00:00Z", 1.4, 1.1,
                         probabilities, result)  # fmt: skip


# a calibrated model that tells the matches apart: the matches it rates 0.6, 0.25, 0.15 end 12
# home wins, 5 draws and 3 away wins in every 20, and those it rates the mirror end the mirror
STRONG, WEAK = (0.6, 0.25, 0.15), (0.15, 0.25, 0.6)
ENDS = ["home"] * 12 + ["draw"] * 5 + ["away"] * 3


def test_a_model_is_judged_against_the_frequencies_of_the_seasons_before():
    rules = make_blind_rules(RESEARCH)
    forecasts = []
    for season in (2014, 2015, 2016, 2017):
        for week in range(4):
            for index, end in enumerate(ENDS):
                mirror = {"home": "away", "away": "home", "draw": "draw"}[end]
                number = season * 1000 + week * 100 + index
                forecasts.append(_forecast(number, season, STRONG, end))
                forecasts.append(_forecast(number + 50, season, WEAK, mirror))
    verdict = judge_against_frequencies(forecasts, (2016, 2017), rules)
    assert verdict.passed is True  # a calibrated model that tells matches apart
    assert "| 320 |" in verdict.rows[2]
    swapped = [_forecast(f.match_id, f.season, WEAK if f.probabilities == STRONG else STRONG,
                         f.result) for f in forecasts]  # fmt: skip
    assert judge_against_frequencies(swapped, (2016, 2017), rules).passed is False
    assert judge_against_frequencies(forecasts, (2030,), rules).passed is None


def test_the_blind_verdicts_run_once_per_commit(tmp_path):
    assert find_earlier_report("abc1234", tmp_path) is None
    (tmp_path / "2026-10-26-blind-verdicts.md").write_text(
        "# The blind verdicts\n\nWorld: `w`. Code: `abc1234`. Closed seasons: 2024, 2025.\n",
        encoding="utf-8")  # fmt: skip
    assert find_earlier_report("abc1234", tmp_path) is not None
    assert find_earlier_report("def5678", tmp_path) is None
