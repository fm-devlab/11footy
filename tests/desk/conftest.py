"""The window over two datasets, each with the sample forecast log beside it
(`tests/fixtures`):
- ours: the small world's whole file, unpacked from its ready release
  (`tests/fixtures/desk/release`, rebuilt by `tests/data/make_desk_fixtures.py`);
- theirs: a hand-made dataset with the format's required columns only, its match to come
  numbered as the small world's, so the same log stands for the one Vesta would write beside it.
"""

import lzma
import shutil
from pathlib import Path

import pytest
from desk_datasets import RELEASE, build_theirs
from sample_log import build as build_log

from desk.reading import find_log


@pytest.fixture(scope="session")
def sample_log(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_log(tmp_path_factory.mktemp("log") / "forecast-log.db")


@pytest.fixture(scope="session")
def ours(tmp_path_factory: pytest.TempPathFactory, sample_log: Path) -> Path:
    """Our dataset as the export writes it, with the sample log beside it."""
    dataset = tmp_path_factory.mktemp("ours") / "11footy.db"
    dataset.write_bytes(lzma.decompress((RELEASE / "11footy.db.xz").read_bytes()))
    shutil.copy(sample_log, find_log(dataset))
    return dataset


@pytest.fixture
def theirs(tmp_path: Path, sample_log: Path) -> Path:
    dataset = build_theirs(tmp_path / "theirs.db")
    shutil.copy(sample_log, find_log(dataset))
    return dataset
