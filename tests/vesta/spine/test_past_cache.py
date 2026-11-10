"""The past the morning keeps: used only by the same release, the same season and the same
inputs; another release's past is never reused."""

from pathlib import Path

import numpy as np

from vesta.jobs.past_cache import (
    KeptPast,
    build_cache_path,
    hash_inputs,
    hash_release,
    read_kept,
    write_kept,
)
from vesta.spine.world import LeagueMatch


def _repository(root: Path) -> Path:
    for folder, name, text in (("src/vesta", "model.py", "A = 1\n"),
                               ("config/vesta", "spine.toml", "a = 1\n")):  # fmt: skip
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / name).write_text(text, encoding="utf-8")
    return root


def test_the_release_changes_with_any_file_of_the_model(tmp_path):
    root = _repository(tmp_path)
    first = hash_release(root)
    (root / "src/vesta/__pycache__").mkdir()
    (root / "src/vesta/__pycache__/model.cpython-312.pyc").write_bytes(b"compiled")
    assert hash_release(root) == first  # compiled files are not the model
    (root / "config/vesta/spine.toml").write_text("a = 2\n", encoding="utf-8")
    assert hash_release(root) != first


def test_a_kept_past_is_used_only_by_its_release_season_and_inputs(tmp_path):
    kept = KeptPast("release-1", 2026, "inputs-1", np.array([0.1, 0.2]),
                    ((1, "2025-08-01T15:00:00Z", 1.4, 1.1),))  # fmt: skip
    write_kept(tmp_path, kept)
    found = read_kept(tmp_path, "release-1", 2026, "inputs-1")
    assert found is not None and found.forecasts == kept.forecasts
    assert read_kept(tmp_path, "release-2", 2026, "inputs-1") is None  # another version
    assert read_kept(tmp_path, "release-1", 2026, "inputs-2") is None  # a corrected world
    assert read_kept(tmp_path, "release-1", 2027, "inputs-1") is None
    assert read_kept(None, "release-1", 2026, "inputs-1") is None
    build_cache_path(tmp_path, 2026).write_bytes(b"not a kept past")
    assert read_kept(tmp_path, "release-1", 2026, "inputs-1") is None


class _Planted:
    """What a hostile file would run when loaded: it writes a marker."""

    marker: Path

    def __reduce__(self):
        return (Path.write_text, (self.marker, "ran"))


def test_a_planted_file_in_the_cache_is_refused_and_never_runs(tmp_path):
    """A dataset folder handed over ready-made may carry a cache file: the kept past is plain
    numbers and text, so a pickled object, as the past's own file or inside its archive, is
    refused and its code never runs."""
    import pickle

    planted = _Planted()
    planted.marker = tmp_path / "marker.txt"
    payload = pickle.dumps(planted)
    path = build_cache_path(tmp_path, 2026)
    path.write_bytes(payload)
    assert read_kept(tmp_path, "release-1", 2026, "inputs-1") is None
    with path.open("wb") as handle:  # an archive whose arrays are pickled objects
        np.savez(handle, release=np.array([planted], dtype=object), season=np.int64(2026))
    assert read_kept(tmp_path, "release-1", 2026, "inputs-1") is None
    (tmp_path / "pre-match-past-2026.pickle").write_bytes(payload)  # an earlier version's name
    assert read_kept(tmp_path, "release-1", 2026, "inputs-1") is None
    assert not planted.marker.exists()


def test_a_new_season_replaces_the_old_past(tmp_path):
    write_kept(tmp_path, KeptPast("r", 2025, "i", np.array([]), ()))
    write_kept(tmp_path, KeptPast("r", 2026, "i", np.array([]), ()))
    (tmp_path / "pre-match-past-2024.pickle").write_bytes(b"an earlier version's file")
    write_kept(tmp_path, KeptPast("r", 2026, "i", np.array([]), ()))
    assert [path.name for path in tmp_path.iterdir()] == ["pre-match-past-2026.npz"]


def test_the_inputs_change_with_a_corrected_result(world):
    from vesta.expected_goals.shots import ShotTable

    match = LeagueMatch(1, "PL", 2025, 100, 101, "2025-08-01T15:00:00Z", "minute",
                        "2025-08-01", 2, 1)  # fmt: skip
    empty = ShotTable(*(np.array([]) for _ in range(16)))
    corrected = LeagueMatch(*[*match.__dict__.values()][:-2], 0, 3)
    assert hash_inputs([match], {}, empty) == hash_inputs([match], {}, empty)
    assert hash_inputs([match], {}, empty) != hash_inputs([corrected], {}, empty)
