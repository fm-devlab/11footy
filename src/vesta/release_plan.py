"""The plan of a release, written from its verdicts: exactly the keys that passed, where.

    python -m vesta.release_plan <development report> <blind report> <version>

reads the two reports (`python -m vesta.verdicts`, `python -m vesta.blind_verdicts`, run on the
same world and code) and writes `config/vesta/releases/<version>.toml`, the plan
`python -m vesta.releases` registers. The rules, by family member:

- the spine's M1-M5 in the leagues of the champion (`[walk_forward] competitions`) when B0 and
  the champion pass (the leagues have no blind verdict: they are measured forward, in public);
- the spine's M1-M5 in the UEFA cups when the cups pass, development and blind; in the national
  teams' competitions when the nations pass, both;
- each goals-in-time key in the leagues when it passes on its own, development and blind.
Every key is pre-match. A verdict that did not run counts as not passed. The season stage's keys
(T2 `.so_far`) and the ties (M14) are not published by any job and stay out.
"""

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPINE_KEYS = ("M1.result", "M2.score", "M3.goals", "M4.both_score", "M5.goals.team")
B0, CHAMPION = "B0 (the previous baseline)", "The champion"
CUPS, NATIONS = "The UEFA cups", "The national teams"
GOALS_IN_TIME = "Goals in time (step 3)"


def parse_results(report: str) -> dict[str, bool]:
    """The summary table: verdict name -> passed (a verdict that did not run: False)."""
    found = {}
    for line in report.splitlines():
        match = re.fullmatch(r"\| (.+?) \| (pass|fail|did not run) \|", line.strip())
        if match:
            found[match.group(1)] = match.group(2) == "pass"
    return found


def parse_passing_keys(report: str, section: str) -> set[str]:
    """The keys of a section's table whose last column says pass."""
    keys, inside = set(), False
    for line in report.splitlines():
        if line.startswith("## "):
            inside = line[3:].strip() == section
        elif inside and line.startswith("| ") and line.rstrip().endswith("| pass |"):
            keys.add(line.split("|")[1].strip())
    return keys


def plan(development: str, blind: str, version: str, leagues: list[str], cups: list[str],
         development_name: str, blind_name: str) -> str:  # fmt: skip
    """The plan file's text."""
    developed, blinded = parse_results(development), parse_results(blind)

    def has_passed_both(name: str) -> bool:
        return developed.get(name, False) and blinded.get(name, False)

    parts = []
    if developed.get(B0, False) and developed.get(CHAMPION, False):
        parts.append(("spine", leagues, sorted(SPINE_KEYS)))
    if has_passed_both(CUPS):
        parts.append(("spine", cups, sorted(SPINE_KEYS)))
    if has_passed_both(NATIONS):
        parts.append(("spine", ["nations"], sorted(SPINE_KEYS)))
    timing = parse_passing_keys(development, GOALS_IN_TIME) & parse_passing_keys(
        blind, GOALS_IN_TIME
    )
    if timing:
        parts.append(("goals in time", leagues, sorted(timing)))
    lines = [f"# The release of vesta-{version}, written by `python -m vesta.release_plan` from",
             f"# {development_name} and {blind_name}: exactly the keys that passed.",
             f'version = "{version}"', f'verdict_id = "{development_name}; {blind_name}"',
             'what_changed = "Vesta v2: the keys whose verdicts passed on the world of the '
             'reports"', ""]  # fmt: skip
    for member, competitions, keys in parts:
        lines += ["[[part]]", f'family_member = "{member}"',
                  "competitions = [" + ", ".join(f'"{code}"' for code in competitions) + "]",
                  "keys = [" + ", ".join(f'"{key}"' for key in keys) + "]", ""]  # fmt: skip
    return "\n".join(lines)


def main(arguments: list[str]) -> int:
    if len(arguments) != 3:
        print(__doc__)
        return 2
    development, blind, version = Path(arguments[0]), Path(arguments[1]), arguments[2]
    spine = tomllib.loads((ROOT / "config/vesta/spine.toml").read_text(encoding="utf-8"))
    text = plan(development.read_text(encoding="utf-8"), blind.read_text(encoding="utf-8"),
                version, list(spine["walk_forward"]["competitions"]), list(spine["cups"]["cups"]),
                development.name, blind.name)  # fmt: skip
    path = ROOT / "config/vesta/releases" / f"{version}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"written {path}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
