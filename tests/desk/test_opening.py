import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest
from desk_datasets import RELEASE, build_theirs

from desk import opening
from desk.opening import SUMS, Refused, open_paths

NOW = datetime(2026, 10, 24, tzinfo=UTC)
LEAGUE = "england.premier-league.db.xz"
CUP = "european-cups.champions-league.db.xz"


@pytest.fixture
def release(tmp_path) -> Path:
    """Our dataset as a release (`python -m data seal`): the split files, the whole file
    and SHA256SUMS in one folder, copied from the ready one (`tests/fixtures/desk/release`)."""
    return Path(shutil.copytree(RELEASE, tmp_path / "download"))


def test_the_whole_file_is_unpacked_and_checked(release, ours):
    seen = []
    whole = release / "11footy.db.xz"
    opened = open_paths([whole], lambda doing, done, total: seen.append(doing), now=NOW)
    assert opened.paths == [(release / "11footy.db").resolve()]
    assert opened.data.read_bytes() == ours.read_bytes()
    assert opened.ours and opened.notes == []
    assert opened.cache is None and opened.log.name == "11footy.db.forecast-log.db"
    assert any(doing.startswith("Unpacking") for doing in seen)
    # opened again, the file beside it is used, not unpacked a second time
    written = opened.data.stat().st_mtime_ns
    assert open_paths([whole], now=NOW).data.stat().st_mtime_ns == written


def test_a_release_folder_opens_its_whole_file(release):
    opened = open_paths([release], now=NOW)
    assert [path.name for path in opened.paths] == ["11footy.db"]
    assert any("whole dataset" in note for note in opened.notes)


def test_chosen_split_files_take_the_players_and_join(release):
    opened = open_paths([release / LEAGUE, release / CUP], now=NOW)
    parts = sorted(item.part for item in opened.files)
    assert parts == ["england/premier-league", "european-cups/champions-league", "players"]
    assert all(item.listed for item in opened.files)
    assert opened.cache == release / "vesta" / "cache"
    assert opened.log == release / "vesta" / "forecast-log.db"
    assert opened.data.parent == opened.cache, "the screens read Vesta's joined copy"
    by_code = {item.code: item for item in opened.coverage}
    assert by_code["PL"].part == "england/premier-league"
    assert "La Liga" in by_code["CL"].missing
    assert "Premier League" not in by_code["CL"].missing


def test_the_whole_file_with_split_files_is_refused(release):
    with pytest.raises(Refused, match="twice"):
        open_paths([release / "11footy.db.xz", release / LEAGUE], now=NOW)


def test_a_damaged_file_is_refused(release):
    sums = release / SUMS
    lines = sums.read_text().splitlines(keepends=True)
    sums.write_text("".join("0" * 8 + line[8:] if line.rstrip().endswith(LEAGUE) else line
                            for line in lines))  # fmt: skip
    with pytest.raises(Refused, match="does not match"):
        open_paths([release / LEAGUE], now=NOW)


def test_our_file_without_its_digests_says_so(ours, tmp_path):
    copy = Path(shutil.copy(ours, tmp_path / ours.name))
    opened = open_paths([copy], now=NOW)
    assert any("Could not confirm" in note for note in opened.notes)


def test_an_old_dataset_says_its_date(tmp_path):
    late = datetime(2026, 12, 1, tzinfo=UTC)
    opened = open_paths([build_theirs(tmp_path / "theirs.db")], now=late)
    assert not opened.ours
    assert any("built on 2026-10-20" in note for note in opened.notes)


def test_a_file_not_in_the_format_is_refused(tmp_path):
    other = tmp_path / "other.db"
    other.write_bytes(b"")
    with pytest.raises(Refused, match="not a dataset in the Vesta data format"):
        open_paths([other])
    with pytest.raises(Refused, match="No file"):
        open_paths([tmp_path / "missing.db"])


def test_unpacking_wants_free_space(release, monkeypatch):
    monkeypatch.setattr(opening.shutil, "disk_usage", lambda _: shutil._ntuple_diskusage(1, 0, 1))
    with pytest.raises(Refused, match="free disk space"):
        open_paths([release / LEAGUE])
    assert not (release / LEAGUE).with_suffix("").exists()


def test_a_whole_file_from_before_parts_holds_its_players(ours, tmp_path):
    """The whole file of an older release names no part: it is still the whole dataset, with
    the players' names in it, so no players file is asked for."""
    copy = Path(shutil.copy(ours, tmp_path / ours.name))
    with sqlite3.connect(copy) as connection:
        connection.execute("alter table format_version drop column part")
    connection.close()
    opened = open_paths([copy], now=NOW)
    assert [item.part for item in opened.files] == ["all"]
    assert opened.files[0].players
    assert not any("players" in note for note in opened.notes)


def test_a_split_file_alone_asks_for_the_players(release, tmp_path):
    alone = Path(shutil.copy(release / LEAGUE, tmp_path / LEAGUE))
    opened = open_paths([alone], now=NOW, check=False)
    assert not opened.files[0].players
    assert any("No players file" in note for note in opened.notes)


def test_a_folder_s_forecast_logs_are_not_datasets(release):
    """A file opened alone leaves its log beside it; the folder opened later skips it."""
    open_paths([release / LEAGUE], now=NOW)
    (release / "england.premier-league.db.forecast-log.db").write_bytes(b"not a dataset")
    (release / "old").mkdir()
    (release / "old" / "forecast-log.db").write_bytes(b"not a dataset")
    found = opening.find_dataset_files([release])
    assert not any("forecast-log" in path.name for path in found)
