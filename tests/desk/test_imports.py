"""Opening the window and running Vesta on a dataset never load LightGBM: only fitting the
expected-goals model does, and on macOS LightGBM needs the system's OpenMP."""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def test_the_window_and_the_run_load_no_lightgbm():
    code = ("import sys; import desk.window, desk.run, vesta.run, vesta.jobs.fixtures; "
            "print('lightgbm' in sys.modules)")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                            check=True)
    assert result.stdout.strip() == "False"
