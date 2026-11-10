"""A dataset in several files: each file opens
alone, a set merges into one working copy, shared rows must agree, a match lives in one file."""

import sqlite3

import pytest

from vesta.data_format import VERSION, create_statements
from vesta.datasets import MergeError, find_dataset_paths, open_datasets
from vesta.spine.world import read_club_matches


def _file(path, competition, matches, built_at="2026-11-08T00:00:00Z", team_name="B"):
    """A file holding one competition (id 1 league XL, id 2 cup XC) and its matches; both files
    carry the two teams they share."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    for statement in create_statements("B7"):
        connection.execute(statement)
    connection.execute("INSERT INTO format_version VALUES (?, '11footy', ?)", (VERSION, built_at))
    code, kind = {1: ("XL", "league"), 2: ("XC", "cup")}[competition]
    connection.execute("INSERT INTO competitions VALUES (?, ?, ?, 'Europe/Rome')",
                       (competition, code, kind))  # fmt: skip
    connection.execute("INSERT INTO seasons VALUES (?, ?, 2025, 'open')",
                       (competition, competition))
    connection.executemany("INSERT INTO teams VALUES (?, 'club', ?)", [(1, "A"), (2, team_name)])
    for match_id, day in matches:
        connection.execute(
            "INSERT INTO matches (match_id, season_id, kickoff_utc, kickoff_precision, local_date,"
            " home_team_id, away_team_id, neutral_venue, status, home_goals, away_goals,"
            " extra_time_played) VALUES (?, ?, ?, 'minute', ?, 1, 2, 0, 'played', 1, 0, 0)",
            (match_id, competition, f"2025-09-{day}T18:00:00Z", f"2025-09-{day}"))  # fmt: skip
    connection.commit()
    connection.close()
    return path


def test_a_set_of_split_files_reads_as_one_dataset(tmp_path):
    league = _file(tmp_path / "data/italy/serie-a.db", 1, [(1, "01"), (2, "08")])
    cup = _file(tmp_path / "data/european-cups/cl.db", 2, [(3, "03")],
                built_at="2026-11-07T00:00:00Z")  # fmt: skip
    paths = find_dataset_paths([tmp_path / "data"])
    assert paths == sorted([league.resolve(), cup.resolve()])
    connection, files = open_datasets(paths, tmp_path / "cache")
    assert [m.match_id for m in read_club_matches(connection, ["XL"], ["XC"], 2025, 2025,
                                                  2025)] == [1, 3, 2]  # fmt: skip
    assert connection.execute("SELECT count(*) FROM teams").fetchone()[0] == 2  # shared once
    assert connection.execute("SELECT dataset, built_at FROM format_version").fetchone() == (
        "11footy", "2026-11-07T00:00:00Z")  # the oldest state of the set
    assert len({item.sha256 for item in files}) == 2
    connection.close()
    again, _ = open_datasets(paths, tmp_path / "cache")  # the working copy is kept
    assert len(list((tmp_path / "cache").glob("merged-*.db"))) == 1
    again.close()


def test_one_file_opens_as_it_is(tmp_path):
    league = _file(tmp_path / "serie-a.db", 1, [(1, "01")])
    connection, files = open_datasets(find_dataset_paths([league]), tmp_path / "cache")
    assert connection.execute("SELECT count(*) FROM matches").fetchone()[0] == 1
    assert not (tmp_path / "cache").exists()
    connection.close()


def test_a_match_in_two_files_is_refused(tmp_path):
    whole = _file(tmp_path / "all.db", 1, [(1, "01"), (2, "08")])
    split = _file(tmp_path / "serie-a.db", 1, [(1, "01")])
    with pytest.raises(MergeError, match="lives in one file"):
        open_datasets(find_dataset_paths([whole, split]), tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.db"))  # nothing half-made is kept


def test_shared_rows_that_disagree_are_refused(tmp_path):
    league = _file(tmp_path / "serie-a.db", 1, [(1, "01")])
    cup = _file(tmp_path / "cl.db", 2, [(3, "03")], team_name="B renamed")
    with pytest.raises(MergeError, match="teams"):
        open_datasets(find_dataset_paths([league, cup]), tmp_path / "cache")



def test_a_hostile_tables_own_sql_is_never_run(tmp_path):
    """A table whose CREATE statement carries a default that loads an extension and a check is
    created again by our own statement: none of its SQL is kept or run, its rows come."""
    league = _file(tmp_path / "serie-a.db", 1, [(1, "01")])
    cup = _file(tmp_path / "cl.db", 2, [(3, "03")])
    connection = sqlite3.connect(cup)
    connection.execute("CREATE TABLE notes (note_id INTEGER PRIMARY KEY, text TEXT"
                       " DEFAULT (load_extension('evil')) CHECK (length(text) > 99))")  # fmt: skip
    connection.execute("INSERT INTO notes (note_id, text) VALUES (1, " + "'x' || "
                       "printf('%.99c', 'y'))")  # fmt: skip
    connection.commit()
    connection.close()
    merged, _ = open_datasets(find_dataset_paths([league, cup]), tmp_path / "cache")
    statement = merged.execute("SELECT sql FROM sqlite_master WHERE name = 'notes'").fetchone()[0]
    assert "load_extension" not in statement and "CHECK" not in statement
    assert merged.execute("SELECT note_id FROM notes").fetchall() == [(1,)]
    merged.close()


@pytest.mark.parametrize("statement", [
    'CREATE TABLE "teams; DROP TABLE matches" (team_id INTEGER)',
    'CREATE TABLE odd ("name) VALUES (1); --" TEXT)',
    "CREATE TABLE typed (value TEXT COLLATE nocase, other WEIRD_TYPE)",
])
def test_a_table_or_column_name_or_type_outside_the_plain_ones_refuses_the_set(tmp_path,
                                                                                statement):
    league = _file(tmp_path / "serie-a.db", 1, [(1, "01")])
    cup = _file(tmp_path / "cl.db", 2, [(3, "03")])
    connection = sqlite3.connect(cup)
    connection.execute(statement)
    connection.commit()
    connection.close()
    with pytest.raises(MergeError, match="plain identifier|declares the type"):
        open_datasets(find_dataset_paths([league, cup]), tmp_path / "cache")
    assert not list((tmp_path / "cache").glob("*.db"))


def _signed_release(folder, files):
    """SHA256SUMS of `files` signed with a fresh ed25519 key; the allowed signers file."""
    import shutil
    import subprocess

    from vesta.datasets import hash_file

    program = shutil.which("ssh-keygen")
    if program is None:
        pytest.skip("ssh-keygen is not installed")
    lines = [f"{hash_file(path)}  {path.relative_to(folder).as_posix()}" for path in files]
    (folder / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    key = folder.parent / "key"
    run = {"stdin": subprocess.DEVNULL, "capture_output": True, "check": True, "timeout": 60}
    subprocess.run([program, "-q", "-t", "ed25519", "-N", "", "-C", "test", "-f", str(key)],
                   **run)  # fmt: skip
    subprocess.run([program, "-Y", "sign", "-f", str(key), "-n", "11footy-dataset",
                    str(folder / "SHA256SUMS")], **run)  # fmt: skip
    signers = folder.parent / "signers"
    public = key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    signers.write_text(f'11footy-dataset namespaces="11footy-dataset" {public}\n',
                       encoding="utf-8")  # fmt: skip
    return signers


def test_a_file_is_ours_only_by_its_hash_in_the_signed_sums(tmp_path):
    from vesta.dataset_proof import prove
    from vesta.datasets import describe

    folder = tmp_path / "release"
    league = _file(folder / "italy/serie-a.db", 1, [(1, "01")])
    cup = _file(folder / "european-cups/cl.db", 2, [(3, "03")])
    signers = _signed_release(folder, [league, cup])
    assert prove(describe(league), signers).ours and prove(describe(cup), signers).ours
    # a file changed after the release: its hash is no longer listed
    connection = sqlite3.connect(cup)
    connection.execute("UPDATE matches SET home_goals = 5")
    connection.commit()
    connection.close()
    assert prove(describe(cup), signers).level == "none"
    # the sums rewritten to list the changed file: the signature no longer verifies
    from vesta.datasets import hash_file

    (folder / "SHA256SUMS").write_text(
        f"{hash_file(league)}  italy/serie-a.db\n{hash_file(cup)}  european-cups/cl.db\n",
        encoding="utf-8")  # fmt: skip
    changed = prove(describe(cup), signers)
    assert not changed.ours and changed.level == "listed"
    # no key shipped: nothing is ours
    empty = tmp_path / "no-signers"
    empty.write_text("# no key yet\n", encoding="utf-8")
    assert prove(describe(league), empty).level == "listed"


def _players_file(path):
    """The players file of a split release: every player, no match; its scope says so."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE format_version (version TEXT, dataset TEXT, built_at TEXT,"
                       " scope TEXT)")  # fmt: skip
    connection.execute("INSERT INTO format_version VALUES (?, '11footy',"
                       " '2026-11-08T00:00:00Z', 'players')", (VERSION,))  # fmt: skip
    connection.execute("CREATE TABLE players (player_id INTEGER PRIMARY KEY, name TEXT)")
    connection.executemany("INSERT INTO players VALUES (?, ?)", [(10, "One"), (11, "Two")])
    connection.commit()
    connection.close()
    return path


def test_a_set_with_the_players_file_runs(tmp_path):
    from vesta.datasets import describe
    from vesta.run import main

    _file(tmp_path / "data/italy/serie-a.db", 1, [(1, "01"), (2, "08")])
    _file(tmp_path / "data/european-cups/champions-league.db", 2, [(3, "03")])
    players = _players_file(tmp_path / "data/players.db")
    assert describe(players).codes == frozenset()
    connection, files = open_datasets(find_dataset_paths([tmp_path / "data"]), tmp_path / "cache")
    assert connection.execute("SELECT count(*) FROM players").fetchone()[0] == 2
    assert connection.execute("SELECT count(*) FROM matches").fetchone()[0] == 3
    connection.close()
    assert len(files) == 3
    assert main([str(tmp_path / "data"), "--now", "2025-09-02T00:00:00Z"]) == 0
