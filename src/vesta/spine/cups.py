"""The spine for the UEFA club cups: one Elo across leagues and cups, then a Poisson.

The reference walk-forward of 2019-2026: 0.9557 on 6,069 matches. A cup match needs a strength
for clubs the leagues never meet, on the same scale as the clubs they know:

1. one Elo rating over every match of the five leagues since 2001-02 and of the cups since
   2014-15, in kick-off order, so the cups tie the leagues to the rest of Europe. A club of the
   five leagues starts at `first_rating_of_a_league_club` in the first league season, or, once
   promoted, at the mean of the lowest ratings of its league; any other club at the mean rating of
   its country's clubs (its country: the most common country of the venues of its home cup
   matches), or at `first_rating_of_another_club` when its country has no rated club yet;
2. for each side, goals ~ Poisson with
       log intensity = alpha + (beta + steeper * q) * difference / 400 + gamma * venue + qual * q
   (q = 1 in a qualifying round, where the gaps are wider; venue +1 at home, 0 on neutral ground,
   the away side with the signs of the difference and the venue turned), fitted on the cup matches
   of the seasons before the one forecast, then the spine's matrix calibration on the same matches.

The candidate "cups v3" (its plan of 1 November 2026) adds three
switches of `[cups]`, all off in the released model: `venue_by_phase` (the venue term
`(gamma + gamma_q * q) * venue`), `second_leg_state` (in a second leg, each side's log intensity
gains `kappa * lead`, its lead on aggregate after the first leg's 90 minutes clipped to +/-2) and
`calibration_spread` (the matrix calibration with its spread and dispersion).
"""

from collections import Counter, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from vesta.determinism import run_on_one_thread
from vesta.spine.calibration import fit_calibration
from vesta.spine.elo import ELO_SCALE, compute_rating_change
from vesta.spine.match_forecast import RESULTS, MatchForecast
from vesta.spine.score_matrix import Calibration, score_matrices, sum_result_probabilities

QUALIFYING_WORDS = ("qualifying", "preliminary")
SUMMER_PLAY_OFFS = ("playoff round", "play-off round", "playoffs", "playoff")
SUMMER_MONTHS = ("06", "07", "08", "09", "10")
SWITCHES = ("venue_by_phase", "second_leg_state", "calibration_spread")
# a second leg's lead on aggregate is clipped to this many goals either way
LEAD_CLIP = 2


@dataclass(frozen=True)
class CupsParameters:
    """`[cups]` of `config/vesta/spine.toml`."""

    cups: tuple[str, ...]
    leagues: tuple[str, ...]
    first_league_season: int
    first_cup_season: int
    weight: float
    home_advantage: float
    first_rating_of_a_league_club: float
    first_rating_of_another_club: float
    promoted_peers: int
    # `[time] result_known_after_minutes`: a club's first rating reads its peers' ratings as of
    # the results known by the kick-off (the known-before check of family A, 31 October 2026)
    result_known_after_minutes: int = 135
    # the Elo's result: "ninety" (after 90 minutes) or "final" (extra time included)
    elo_on: str = "final"
    # the candidate "cups v3"'s switches (the module's docstring), off in the released model
    venue_by_phase: bool = False
    second_leg_state: bool = False
    calibration_spread: bool = False

    @classmethod
    def from_config(cls, spine: dict) -> "CupsParameters":
        section = spine["cups"]
        return cls(**{
            name: tuple(section[name]) if isinstance(section[name], list) else section[name]
            for name in cls.__dataclass_fields__
            if name not in ("result_known_after_minutes", "elo_on", *SWITCHES)
        }, result_known_after_minutes=int(spine["time"]["result_known_after_minutes"]),
            elo_on=str(section.get("elo_on", "final")),
            **{name: bool(section.get(name, False)) for name in SWITCHES})  # fmt: skip


@dataclass(frozen=True)
class ClubMatch:
    """A finished match of a league or a cup, as the rating reads it."""

    match_id: int
    competition: str
    competition_kind: str
    season: int
    stage: str
    kickoff_utc: str
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    neutral: bool
    venue_country: str | None
    # the final score, extra time included (the goals above are after 90 minutes): the Elo of
    # `[cups] elo_on = "final"` reads it
    final_home_goals: int | None = None
    final_away_goals: int | None = None
    # the leg of a two-legged tie (1 or 2) and the tie, None elsewhere
    leg: int | None = None
    tie_id: int | None = None

    @property
    def is_cup(self) -> bool:
        return self.competition_kind in ("cup", "cup qualifying", "super cup")

    @property
    def qualifying(self) -> bool:
        """A qualifying round: a qualifying competition, a stage named so, or a play-off played
        in summer before the main stage (the knockout play-offs of February are main stage)."""
        stage = self.stage.lower()
        return (
            self.competition_kind == "cup qualifying"
            or any(word in stage for word in QUALIFYING_WORDS)
            or (stage in SUMMER_PLAY_OFFS and self.kickoff_utc[5:7] in SUMMER_MONTHS)
        )


@dataclass(frozen=True)
class RatedCupMatch:
    match: ClubMatch
    home_rating: float
    away_rating: float
    # in a second leg, the home side's lead on aggregate after the first leg's 90 minutes, clipped
    # to +/- `LEAD_CLIP` (0 elsewhere, and where the first leg is not known)
    lead: float = 0.0


def rate_matches(matches: list[ClubMatch], parameters: CupsParameters) -> list[RatedCupMatch]:
    """Every cup match with the two ratings before its kick-off; the leagues feed the ratings."""
    return _rate(matches, parameters)[0]


def make_rating_lookup(matches: list[ClubMatch], parameters: CupsParameters):
    """A function giving each club's rating after all of `matches` (a club never met gets its
    first rating by the same rule as in the rating)."""
    _, rating, first = _rate(matches, parameters)

    def look_up_rating(team: int) -> float:
        return rating[team] if team in rating else first(team, rating)

    return look_up_rating


def _rate(matches, parameters):
    ordered = sorted(matches, key=lambda match: (match.kickoff_utc, match.is_cup, match.match_id))
    league_of, first_league_season = _find_league_clubs(ordered, parameters)
    country = _find_club_countries(ordered)
    rating: dict[int, float] = {}
    # the ratings as of the results known by the kick-off being rated: a club new to the rating
    # starts from its peers' known ratings, never from a result still being played
    known: dict[int, float] = {}
    pending: deque[tuple[str, int, float]] = deque()  # (known at, club, rating after)
    delay = timedelta(minutes=parameters.result_known_after_minutes)
    first_legs: dict[int, ClubMatch] = {}
    rated = []
    for match in ordered:
        while pending and pending[0][0] <= match.kickoff_utc:
            _, club, value = pending.popleft()
            known[club] = value
        for team in (match.home_team_id, match.away_team_id):
            if team not in rating:
                rating[team] = _choose_first_rating(team, known, league_of, first_league_season,
                                                    country, parameters)  # fmt: skip
                known.setdefault(team, rating[team])
        home_rating, away_rating = rating[match.home_team_id], rating[match.away_team_id]
        if match.is_cup:
            rated.append(RatedCupMatch(match, home_rating, away_rating,
                                       find_second_leg_lead(match, first_legs, delay)))  # fmt: skip
            if match.leg == 1 and match.tie_id is not None:
                first_legs[match.tie_id] = match
        home_goals, away_goals = match.home_goals, match.away_goals
        if parameters.elo_on == "final" and match.final_home_goals is not None:
            home_goals, away_goals = match.final_home_goals, match.final_away_goals
        change = compute_rating_change(
            home_rating, away_rating, home_goals, away_goals, parameters.weight,
            0.0 if match.neutral else parameters.home_advantage,
        )  # fmt: skip
        rating[match.home_team_id] = home_rating + change
        rating[match.away_team_id] = away_rating - change
        known_at = _add_delay(match.kickoff_utc, delay)
        pending.append((known_at, match.home_team_id, rating[match.home_team_id]))
        pending.append((known_at, match.away_team_id, rating[match.away_team_id]))

    def rate_unrated_club(team: int, known: dict[int, float]) -> float:
        return _choose_first_rating(
            team, known, league_of, first_league_season, country, parameters
        )

    return rated, rating, rate_unrated_club


@dataclass(frozen=True)
class CupsMapping:
    alpha: float
    beta: float
    gamma: float
    qualifying_level: float
    qualifying_slope: float
    calibration: Calibration
    # the candidate "cups v3"'s terms, 0 in the released model
    qualifying_venue: float = 0.0
    lead_slope: float = 0.0

    def predict_intensities(self, rows: list[RatedCupMatch]) -> tuple[np.ndarray, np.ndarray]:
        difference, venue, qualifying = _build_design(rows)
        lead = np.array([row.lead for row in rows])
        slope = self.beta + self.qualifying_slope * qualifying
        common = self.alpha + self.qualifying_level * qualifying
        ground = (self.gamma + self.qualifying_venue * qualifying) * venue
        home = np.exp(common + slope * difference + ground + self.lead_slope * lead)
        away = np.exp(common - slope * difference - ground - self.lead_slope * lead)
        return home, away


@run_on_one_thread
def fit_mapping(rows: list[RatedCupMatch], parameters: CupsParameters | None = None) -> CupsMapping:
    """The Poisson of both sides' goals, then the matrix's calibration on the same matches; the
    candidate's terms and calibration as `parameters`' switches say (none without)."""
    venue_by_phase = parameters is not None and parameters.venue_by_phase
    second_leg_state = parameters is not None and parameters.second_leg_state
    spread = parameters is not None and parameters.calibration_spread
    difference, venue, qualifying = _build_design(rows)
    lead = np.array([row.lead for row in rows])
    home_goals = np.array([row.match.home_goals for row in rows], dtype="float64")
    away_goals = np.array([row.match.away_goals for row in rows], dtype="float64")
    differences = np.concatenate([difference, -difference])
    venues = np.concatenate([venue, -venue])
    qualifyings = np.concatenate([qualifying, qualifying])
    leads = np.concatenate([lead, -lead])
    goals = np.concatenate([home_goals, away_goals])
    log_factorial = gammaln(goals + 1)
    extra = [name for name, on in (("qualifying_venue", venue_by_phase),
                                   ("lead_slope", second_leg_state)) if on]  # fmt: skip

    def measure_negative_log_likelihood(theta: np.ndarray) -> float:
        terms = dict(zip(extra, theta[5:], strict=True))
        linear = (theta[0] + (theta[1] + theta[4] * qualifyings) * differences
                  + (theta[2] + terms.get("qualifying_venue", 0.0) * qualifyings) * venues
                  + theta[3] * qualifyings + terms.get("lead_slope", 0.0) * leads)  # fmt: skip
        return float((np.exp(linear) - goals * linear + log_factorial).sum())

    start = np.array([0.3, 0.5, 0.1, 0.0, 0.0] + [0.0] * len(extra))
    fitted = minimize(measure_negative_log_likelihood, start, method="L-BFGS-B")
    alpha, beta, gamma, level, slope = (float(value) for value in fitted.x[:5])
    terms = {name: float(value) for name, value in zip(extra, fitted.x[5:], strict=True)}
    uncalibrated = CupsMapping(alpha, beta, gamma, level, slope, Calibration(), **terms)
    home, away = uncalibrated.predict_intensities(rows)
    calibration = fit_calibration(home, away, home_goals, away_goals, spread, spread)
    return CupsMapping(alpha, beta, gamma, level, slope, calibration, **terms)


def find_second_leg_lead(match: ClubMatch, first_legs: dict[int, ClubMatch],
                         delay: timedelta) -> float:  # fmt: skip
    """A second leg's home side's lead on aggregate after the first leg's 90 minutes, clipped to
    +/- `LEAD_CLIP`; 0 for any other match and for a second leg whose first leg is not known. The
    first leg's result must be known before the second leg's kick-off."""
    if match.leg != 2 or match.tie_id is None or match.tie_id not in first_legs:
        return 0.0
    first = first_legs[match.tie_id]
    if _add_delay(first.kickoff_utc, delay) > match.kickoff_utc:
        raise ValueError(f"the first leg of tie {match.tie_id} is not known before the second"
                         f" leg {match.match_id}")  # fmt: skip
    if first.home_team_id == match.home_team_id:
        lead = first.home_goals - first.away_goals
    else:
        lead = first.away_goals - first.home_goals
    return float(max(-LEAD_CLIP, min(LEAD_CLIP, lead)))


def forecast_cups(
    matches: list[ClubMatch], parameters: CupsParameters, seasons: tuple[int, ...]
) -> list[MatchForecast]:
    """Each cup season forecast by the mapping fitted on the cup matches of the seasons before."""
    rated = [row for row in rate_matches(matches, parameters)
             if row.match.competition in parameters.cups]  # fmt: skip
    forecasts = []
    for season in seasons:
        training = [row for row in rated
                    if parameters.first_cup_season <= row.match.season < season]  # fmt: skip
        forecast = [row for row in rated if row.match.season == season]
        if not training or not forecast:
            continue
        mapping = fit_mapping(training, parameters)
        home, away = mapping.predict_intensities(forecast)
        probabilities = sum_result_probabilities(score_matrices(home, away, mapping.calibration))
        for index, row in enumerate(forecast):
            forecasts.append(MatchForecast(
                match_id=row.match.match_id, competition=row.match.competition, season=season,
                forecast_time=row.match.kickoff_utc, home_intensity=float(home[index]),
                away_intensity=float(away[index]),
                probabilities=tuple(float(p) for p in probabilities[index]),
                result=_classify_result(row.match), calibration=mapping.calibration,
            ))  # fmt: skip
    return forecasts


def _find_league_clubs(
    matches: list[ClubMatch], parameters: CupsParameters
) -> tuple[dict[int, str], dict[int, int]]:
    """The league of each club of the five leagues, and its first league season."""
    league_of: dict[int, str] = {}
    first: dict[int, int] = {}
    for match in matches:
        if match.competition in parameters.leagues:
            for team in (match.home_team_id, match.away_team_id):
                league_of.setdefault(team, match.competition)
                first[team] = min(first.get(team, match.season), match.season)
    return league_of, first


def _find_club_countries(matches: list[ClubMatch]) -> dict[int, str]:
    """The country of a club: the most common country of the venues of its home cup matches."""
    seen: dict[int, Counter] = {}
    for match in matches:
        if match.is_cup and not match.neutral and match.venue_country:
            seen.setdefault(match.home_team_id, Counter())[match.venue_country] += 1
    return {team: counts.most_common(1)[0][0] for team, counts in seen.items()}


def _add_delay(kickoff_utc: str, delay: timedelta) -> str:
    moment = datetime.strptime(kickoff_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return (moment + delay).strftime("%Y-%m-%dT%H:%M:%SZ")


def _choose_first_rating(
    team: int,
    rating: dict[int, float],
    league_of: dict[int, str],
    first_league_season: dict[int, int],
    country: dict[int, str],
    parameters: CupsParameters,
) -> float:
    if team in league_of:
        if first_league_season[team] <= parameters.first_league_season:
            return parameters.first_rating_of_a_league_club
        peers = sorted(value for club, value in rating.items()
                       if league_of.get(club) == league_of[team])  # fmt: skip
        if len(peers) >= parameters.promoted_peers:
            return float(np.mean(peers[: parameters.promoted_peers]))
        return parameters.first_rating_of_a_league_club
    nation = country.get(team)
    if nation is not None:
        peers = [value for club, value in rating.items()
                 if club not in league_of and country.get(club) == nation]  # fmt: skip
        if peers:
            return float(np.mean(peers))
    return parameters.first_rating_of_another_club


def _build_design(rows: list[RatedCupMatch]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = np.array([(row.home_rating - row.away_rating) / ELO_SCALE for row in rows])
    venue = np.array([0.0 if row.match.neutral else 1.0 for row in rows])
    qualifying = np.array([1.0 if row.match.qualifying else 0.0 for row in rows])
    return difference, venue, qualifying


def _classify_result(match: ClubMatch) -> str:
    if match.home_goals > match.away_goals:
        return RESULTS[0]
    return RESULTS[1] if match.home_goals == match.away_goals else RESULTS[2]
