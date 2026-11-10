"""The lineup-time candidate of the spine (the lineup-time plan of 2 November 2026, part 2;
hypothesis H4): the player ratings read the eleven announced for the match itself instead of the
eleven that started each side's last match. Everything else is the champion's: the same weekly
fit, the same team model, the same blend, the same calibration path. Champion and candidate are
both forecast at kick-off minus 60 minutes, when the sheet is out, so the only difference is the
eleven read. Historically the announced eleven is the world's starting sheet of the match
(`match_players.started`), what the sheet carries at kick-off minus 60 minutes.
"""

from datetime import timedelta

from vesta.spine.player_ratings import PlayerRatingsFamily, PlayerRatingsParameters, RatingsState
from vesta.spine.point_in_time import ForecastRequest

SHEET_OUT = timedelta(minutes=60)  # the sheet is announced about an hour before kick-off
Elevens = dict[int, tuple[tuple[int, ...], tuple[int, ...]]]  # match -> (home, away) starters


def find_announced_elevens(match_facts: dict) -> Elevens:
    """Per match, the starters of each side (home, away), from who played in it."""
    found = {}
    for match_id, facts in match_facts.items():
        sides = tuple(tuple(player for side_of, player, minutes, started, _ in facts.players
                            if side_of == side and started and minutes > 0)
                      for side in (0, 1))  # fmt: skip
        if all(sides):
            found[match_id] = sides
    return found


class AnnouncedElevenFamily(PlayerRatingsFamily):
    """The champion's player ratings, each side rated by its announced eleven when the match has
    one (its last eleven otherwise, as the champion)."""

    def __init__(self, parameters: PlayerRatingsParameters, leagues: tuple[str, ...],
                 announced: Elevens) -> None:  # fmt: skip
        super().__init__(parameters, leagues)
        self.announced = announced

    def choose_eleven(self, state: RatingsState, request: ForecastRequest, team: int) -> tuple:
        sides = self.announced.get(request.match_id)
        if sides is None:
            return super().choose_eleven(state, request, team)
        return sides[0] if team == request.home_team_id else sides[1]


def count_changed_starters(match_facts: dict, matches: list) -> dict[tuple[int, int], int]:
    """(match, team) -> how many of the team's starters were not among the starters of its
    last match before (the champion's eleven), for the matches with both."""
    announced = find_announced_elevens(match_facts)
    last: dict[int, tuple[int, ...]] = {}
    found = {}
    for match in sorted(matches, key=lambda item: (item.kickoff_utc, item.match_id)):
        sides = announced.get(match.match_id)
        if sides is None:
            continue
        for team, eleven in zip((match.home_team_id, match.away_team_id), sides, strict=True):
            if team in last:
                found[(match.match_id, team)] = len(set(eleven) - set(last[team]))
            last[team] = eleven
    return found
