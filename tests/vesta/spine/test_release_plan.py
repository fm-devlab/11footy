"""The plan of a release holds exactly the keys that passed, development and blind, where their
verdicts cover them; the plan reads back through the release's own reader."""

import sqlite3
import tomllib

from vesta.release_plan import plan

ROWS = "| key | metric | n | verdict |\n|---|---|---|---|\n"


def _report(summary: dict[str, str], timing: dict[str, str]) -> str:
    lines = ["# The verdicts", "", "| verdict | result |", "|---|---|"]
    lines += [f"| {name} | {result} |" for name, result in summary.items()]
    lines += ["", "## Goals in time (step 3)", "", ROWS.strip()]
    lines += [f"| {key} | log_loss | 10 | {result} |" for key, result in timing.items()]
    return "\n".join(lines) + "\n"


SPINE = {"B0 (the previous baseline)": "pass", "The champion": "pass"}


def test_a_key_is_published_where_it_passed_development_and_blind():
    development = _report({**SPINE, "The UEFA cups": "pass", "The national teams": "pass",
                           "Goals in time (step 3)": "fail"},
                          {"M7.first_goal": "pass", "M8.first_goal_time": "pass",
                           "M23.own_goal": "fail"})  # fmt: skip
    blind = _report({"The UEFA cups": "pass", "The national teams": "fail",
                     "Goals in time (step 3)": "fail"},
                    {"M7.first_goal": "pass", "M8.first_goal_time": "fail",
                     "M23.own_goal": "pass"})  # fmt: skip
    parts = tomllib.loads(plan(development, blind, "2.0.0", ["PL", "SA"], ["CL"], "d.md",
                               "b.md"))["part"]  # fmt: skip
    found = [(part["family_member"], part["competitions"], part["keys"]) for part in parts]
    spine_keys = ["M1.result", "M2.score", "M3.goals", "M4.both_score", "M5.goals.team"]
    assert found == [
        ("spine", ["PL", "SA"], spine_keys),
        ("spine", ["CL"], spine_keys),
        ("goals in time", ["PL", "SA"], ["M7.first_goal"]),  # passed both
    ]  # the nations failed blind; M8.first_goal_time failed one of the two


def test_the_plan_reads_back_with_the_worlds_ids(tmp_path):
    from vesta.releases import read_plan

    development = _report(SPINE, {"M7.first_goal": "pass"})
    path = tmp_path / "2.0.0.toml"
    path.write_text(plan(development, development, "2.0.0", ["PL"], ["CL"], "d", "b"),
                    encoding="utf-8")  # fmt: skip
    world = sqlite3.connect(":memory:")
    world.execute("CREATE TABLE competitions (code TEXT, competition_id INTEGER)")
    world.execute("INSERT INTO competitions VALUES ('PL', 1)")
    version, _, _, parts = read_plan(path, world)
    assert version == "2.0.0"
    assert [(part.family_member, part.competition_ids) for part in parts] == [
        ("spine", (1,)), ("goals in time", (1,))]
