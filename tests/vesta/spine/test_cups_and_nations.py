"""The spine for the national teams and the UEFA cups: the Elo, the weights, the first ratings,
the qualifying rounds, the mappings and the walk-forwards, on small fixtures."""

import math
import tomllib
from pathlib import Path

import numpy as np
import pytest

from vesta.spine.cups import ClubMatch, CupsParameters, forecast_cups, rate_matches
from vesta.spine.elo import compute_expected_result, compute_rating_change, weigh_margin
from vesta.spine.match_forecast import average_log_loss
from vesta.spine.nations import NationMatch, NationsParameters, forecast_nations
from vesta.spine.nations import rate_matches as rate_nation_matches

ROOT = Path(__file__).resolve().parents[3]
SPINE = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
NATIONS = NationsParameters.from_config(SPINE)
CUPS = CupsParameters.from_config(SPINE)


def test_the_elo_of_world_football_elo():
    assert compute_expected_result(0.0) == 0.5
    assert compute_expected_result(400.0) == pytest.approx(10 / 11)
    assert [weigh_margin(goals) for goals in (0, 1, 2, 3, 5)] == [1.0, 1.0, 1.5, 1.75, 2.0]
    # equal teams, the home side wins 3-0 at home with the bonus of 100
    change = compute_rating_change(1500.0, 1500.0, 3, 0, 40.0, 100.0)
    assert change == pytest.approx(40.0 * 1.75 * (1 - compute_expected_result(100.0)))


def test_the_weight_of_a_national_match_follows_its_competition():
    assert NATIONS.weight("WC", "nations tournament") == 60.0
    assert NATIONS.weight("WCQ", "nations qualifying") == 40.0
    assert NATIONS.weight("UNL", "nations league") == 40.0
    assert NATIONS.weight("EURO", "nations tournament") == 50.0
    assert NATIONS.weight("FRIENDLY", "friendly") == 20.0
    assert NATIONS.weight("NAT_GULF_CUP", "nations tournament") == 30.0


def nation_match(match_id, date, home, away, goals, competition="FRIENDLY", neutral=False):
    kind = "friendly" if competition == "FRIENDLY" else "nations qualifying"
    return NationMatch(match_id, competition, kind, date, home, away, goals[0], goals[1], neutral)


def test_the_ratings_of_a_match_are_those_of_the_start_of_its_day():
    matches = [
        nation_match(1, "2020-06-01", 1, 2, (2, 0)),
        nation_match(2, "2020-06-01", 1, 3, (1, 1)),  # the same day: before the first one counts
        nation_match(3, "2020-06-05", 1, 2, (0, 0)),
    ]
    rated = {row.match.match_id: row for row in rate_nation_matches(matches, NATIONS)}
    assert rated[1].home_rating == rated[2].home_rating == 1500.0
    first = compute_rating_change(1500.0, 1500.0, 2, 0, 20.0, 100.0)
    second = compute_rating_change(1500.0, 1500.0, 1, 1, 20.0, 100.0)
    assert rated[3].home_rating == pytest.approx(1500.0 + first + second)
    assert rated[3].home_played == 2 and rated[3].away_played == 1


def drawn_goals(state: np.random.RandomState, home: float, away: float) -> tuple[int, int]:
    """A score drawn from two strengths, the home side a little stronger at home."""
    return (int(state.poisson(np.exp(0.3 + home - away))),
            int(state.poisson(np.exp(0.0 + away - home))))  # fmt: skip


def synthetic_nations() -> list[NationMatch]:
    """Twelve teams of fixed strengths playing one another for ten years."""
    state = np.random.RandomState(5)
    strength = state.normal(0, 0.4, 12)
    matches, match_id = [], 0
    for year in range(2008, 2018):
        for day in range(1, 13):
            order = state.permutation(12)
            for pair in range(6):
                home, away = int(order[2 * pair]), int(order[2 * pair + 1])
                match_id += 1
                goals = drawn_goals(state, strength[home], strength[away])
                matches.append(nation_match(match_id, f"{year}-{day:02d}-15", home, away, goals,
                                            competition="WCQ"))  # fmt: skip
    return matches


def test_the_nations_walk_forward_forecasts_each_year_with_the_years_before_it():
    forecasts = forecast_nations(synthetic_nations(), NATIONS, (2016, 2017))
    assert {forecast.season for forecast in forecasts} == {2016, 2017}
    for forecast in forecasts:
        assert sum(forecast.probabilities) == pytest.approx(1.0, abs=1e-12)
    assert average_log_loss(forecasts) < math.log(3)
    assert forecast_nations(synthetic_nations(), NATIONS, (2016, 2017)) == forecasts


def test_a_team_without_ten_earlier_matches_is_not_forecast():
    matches = synthetic_nations()[:30]  # five days: no team has ten matches yet
    assert forecast_nations(matches, NATIONS, (2008,)) == []


def club_match(match_id, competition, season, kickoff, home, away, goals, stage="",
               country=None, neutral=False):  # fmt: skip
    kind = {"PL": "league", "SA": "league", "CL": "cup", "CLQ": "cup qualifying"}[competition]
    return ClubMatch(match_id, competition, kind, season, stage, kickoff, home, away, goals[0],
                     goals[1], neutral, country)  # fmt: skip


@pytest.mark.parametrize(
    ("competition", "stage", "kickoff", "qualifying"),
    [
        ("CLQ", "first round", "2021-07-06T18:00:00Z", True),
        ("CL", "qualifying second round", "2016-07-26T18:00:00Z", True),
        ("CL", "playoff round", "2016-08-16T18:00:00Z", True),
        ("CL", "knockout round playoffs", "2025-02-11T20:00:00Z", False),
        ("CL", "group a", "2016-09-13T18:45:00Z", False),
        ("CL", "final", "2016-05-28T18:45:00Z", False),
    ],
)
def test_the_qualifying_rounds_are_known_by_competition_stage_and_month(
    competition, stage, kickoff, qualifying
):
    match = club_match(1, competition, 2016, kickoff, 1, 2, (1, 0), stage=stage)
    assert match.qualifying is qualifying


def test_the_first_ratings_of_league_and_foreign_clubs():
    matches = [
        club_match(1, "PL", 2001, "2001-08-18T14:00:00Z", 10, 11, (1, 0)),
        club_match(2, "PL", 2001, "2001-08-25T14:00:00Z", 12, 13, (0, 2)),
        club_match(4, "CL", 2002, "2002-09-17T18:45:00Z", 20, 11, (2, 1), country="Norway"),
        club_match(5, "CL", 2002, "2002-09-24T18:45:00Z", 21, 13, (0, 0), country="Norway"),
        club_match(6, "CL", 2002, "2002-10-01T18:45:00Z", 22, 12, (1, 1)),  # no country known
    ]
    rated = {row.match.match_id: row for row in rate_matches(matches, CUPS)}
    assert set(rated) == {4, 5, 6}  # only the cups are forecast; the league feeds the ratings
    assert rated[4].away_rating == pytest.approx(
        1500.0 - compute_rating_change(1500, 1500, 1, 0, 30, 60)
    )
    assert rated[4].home_rating == 1300.0  # Norway has no rated club yet
    assert rated[5].home_rating == pytest.approx(
        1300.0 + compute_rating_change(1300.0, rated[4].away_rating, 2, 1, 30, 60)
    )  # the mean of Norway's rated clubs: club 20 alone, after its match
    assert rated[6].home_rating == 1300.0


def test_a_promoted_club_starts_at_the_mean_of_its_leagues_three_lowest():
    matches = [
        club_match(1, "PL", 2001, "2001-08-18T14:00:00Z", 10, 11, (1, 0)),
        club_match(2, "PL", 2001, "2001-08-25T14:00:00Z", 12, 13, (0, 2)),
        club_match(3, "CL", 2002, "2002-07-20T18:45:00Z", 14, 30, (0, 0)),  # 14 before its debut
        club_match(7, "PL", 2002, "2002-08-17T14:00:00Z", 14, 10, (0, 0)),  # promoted in 2002
    ]
    league = sorted([1500.0 + compute_rating_change(1500, 1500, 1, 0, 30, 60),
                     1500.0 - compute_rating_change(1500, 1500, 1, 0, 30, 60),
                     1500.0 + compute_rating_change(1500, 1500, 0, 2, 30, 60),
                     1500.0 - compute_rating_change(1500, 1500, 0, 2, 30, 60)])  # fmt: skip
    rated = rate_matches(matches, CUPS)
    assert rated[0].home_rating == pytest.approx(np.mean(league[:3]))


def synthetic_cups() -> list[ClubMatch]:
    """A league of eight clubs since 2001 and a cup since 2014 with eight foreign clubs."""
    state = np.random.RandomState(9)
    strength = {club: state.normal(0, 0.4) for club in range(1, 17)}
    matches, match_id = [], 0
    for season in range(2001, 2020):
        for week in range(14):
            pairs = ((1 + week % 8, 1 + (week + 3) % 8), (1 + (week + 5) % 8, 1 + (week + 1) % 8))
            for home, away in pairs:
                if home == away:
                    continue
                match_id += 1
                goals = drawn_goals(state, strength[home], strength[away])
                kickoff = f"{season}-08-{1 + week:02d}T15:00:00Z"
                matches.append(club_match(match_id, "PL", season, kickoff, home, away, goals))
        if season >= 2014:
            for leg in range(30):
                home, away = int(state.randint(1, 17)), int(state.randint(1, 17))
                if home == away:
                    continue
                match_id += 1
                goals = drawn_goals(state, strength[home], strength[away])
                stage = "qualifying second round" if leg < 10 else "group a"
                matches.append(club_match(match_id, "CL", season,
                                          f"{season}-09-{1 + leg % 28:02d}T19:00:00Z", home, away,
                                          goals, stage=stage, country="Elsewhere"))  # fmt: skip
    return matches


def test_the_cups_walk_forward_forecasts_each_season_with_the_seasons_before_it():
    forecasts = forecast_cups(synthetic_cups(), CUPS, (2018, 2019))
    assert {forecast.season for forecast in forecasts} == {2018, 2019}
    assert all(forecast.competition == "CL" for forecast in forecasts)
    for forecast in forecasts:
        assert sum(forecast.probabilities) == pytest.approx(1.0, abs=1e-12)
    assert forecast_cups(synthetic_cups(), CUPS, (2018, 2019)) == forecasts


def test_a_new_clubs_first_rating_reads_only_its_peers_known_results():
    """A compatriot's match kicked off an hour before is still being played: the new club starts
    from that compatriot's rating before it (family A's known-before check, 31 October 2026)."""
    matches = [
        club_match(1, "CLQ", 2017, "2017-06-27T17:00:00Z", 30, 40, (1, 1), country="Malta"),
        club_match(2, "CLQ", 2017, "2017-06-29T17:00:00Z", 30, 41, (3, 0), country="Malta"),
        club_match(3, "CLQ", 2017, "2017-06-29T18:00:00Z", 31, 42, (0, 0), country="Malta"),
        club_match(4, "CLQ", 2017, "2017-06-30T18:00:00Z", 32, 43, (0, 0), country="Malta"),
    ]
    rated = {row.match.match_id: row for row in rate_matches(matches, CUPS)}
    before = rated[2].home_rating  # club 30 before its win of match 2
    after = before + compute_rating_change(before, rated[2].away_rating, 3, 0, 30, 60)
    assert rated[3].home_rating == pytest.approx(before)  # match 2 not over at 18:00
    assert rated[4].home_rating == pytest.approx(
        (
            after
            + rated[3].home_rating
            + compute_rating_change(rated[3].home_rating, rated[3].away_rating, 0, 0, 30, 60)
        )
        / 2
    )  # the next day both results are known


def test_the_elo_reads_the_ninety_minutes_or_the_final_score_as_declared():
    """A 1-1 after 90 minutes won 2-1 in extra time: its outcome is a draw either way; the Elo of
    "ninety" rates the draw, the Elo of "final" the win (the plan of 31 October 2026)."""
    from dataclasses import replace

    tie = replace(club_match(1, "CL", 2019, "2020-02-18T20:00:00Z", 50, 60, (1, 1),
                             country="Spain"), final_home_goals=2, final_away_goals=1)
    after = club_match(2, "CL", 2019, "2020-03-10T20:00:00Z", 50, 70, (0, 0), country="Spain")
    ninety = {row.match.match_id: row
              for row in rate_matches([tie, after], replace(CUPS, elo_on="ninety"))}
    final = {row.match.match_id: row
             for row in rate_matches([tie, after], replace(CUPS, elo_on="final"))}
    start = ninety[1].home_rating
    assert ninety[2].home_rating == pytest.approx(
        start + compute_rating_change(start, ninety[1].away_rating, 1, 1, 30, 60))
    assert final[2].home_rating == pytest.approx(
        start + compute_rating_change(start, final[1].away_rating, 2, 1, 30, 60))
    assert final[2].home_rating > ninety[2].home_rating


def test_a_second_legs_lead_is_the_home_sides_aggregate_after_the_first_legs_90_minutes():
    """Cups v3's second-leg state: the second leg's home side was the first leg's away side, which
    won 3-0 there: its lead is +2 after clipping; a match outside a tie has none."""
    from dataclasses import replace

    first = replace(club_match(1, "CL", 2019, "2020-02-18T20:00:00Z", 60, 50, (0, 3),
                               country="Spain"), leg=1, tie_id=7)  # fmt: skip
    second = replace(club_match(2, "CL", 2019, "2020-03-10T20:00:00Z", 50, 60, (1, 1),
                                country="Spain"), leg=2, tie_id=7)  # fmt: skip
    other = club_match(3, "CL", 2019, "2020-03-11T20:00:00Z", 70, 80, (2, 0), country="Spain")
    rated = {row.match.match_id: row for row in rate_matches([first, second, other], CUPS)}
    assert (rated[1].lead, rated[2].lead, rated[3].lead) == (0.0, 2.0, 0.0)


def test_the_candidates_switches_add_their_terms_and_leave_the_champion_alone():
    from dataclasses import replace

    from vesta.spine.cups import fit_mapping

    rows = [row for row in rate_matches(synthetic_cups(), CUPS) if row.match.competition == "CL"]
    champion = fit_mapping(rows)
    assert fit_mapping(rows, CUPS) == champion  # every switch off: the released model
    assert (champion.qualifying_venue, champion.lead_slope) == (0.0, 0.0)
    venue = fit_mapping(rows, replace(CUPS, venue_by_phase=True))
    assert venue.qualifying_venue != 0.0 and venue.lead_slope == 0.0
    forecasts = forecast_cups(synthetic_cups(), replace(CUPS, venue_by_phase=True,
                                                        calibration_spread=True), (2019,))
    assert all(sum(f.probabilities) == pytest.approx(1.0, abs=1e-12) for f in forecasts)


def test_a_larger_configuration_is_chosen_only_when_lower_by_the_margin():
    from vesta.spine.cups_candidate_verdict import Judged, choose

    def judged(name, model, passed=True):
        return Judged(name, 1, model, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, False, passed, {})

    assert choose([judged("V", 0.9500), judged("V+L", 0.9497), judged("V+L+K", 0.9480)]).name == (
        "V+L+K")  # V+L is within 0.0005 of V; V+L+K is 0.0020 below V
    assert choose([judged("V", 0.9500), judged("V+L", 0.9497)]).name == "V"
    assert choose([judged("V", 0.9500, False), judged("V+L", 0.9510)]).name == "V+L"
    assert choose([judged("V", 0.95, False)]) is None
