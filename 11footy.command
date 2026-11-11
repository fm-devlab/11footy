#!/bin/bash
# Opens the 11footy window (README.md, "Start"). The first time, uv installs
# Python 3.12 and the libraries the window needs: a few minutes, once.
# On macOS double-click it in Finder; the first time right-click it, then Open.
set -u
cd "$(dirname "$0")" || exit 1
if command -v uv >/dev/null 2>&1; then
    UV=uv
else
    UV="$HOME/.local/bin/uv"
fi
if [ "$UV" != uv ] && [ ! -x "$UV" ]; then
    echo 'uv is not installed. Install it first: see README.md, "Start".'
    read -r -p "Press Enter to close. " _
    exit 1
fi
# uv's own Python, which always has Tk: a system Python may not (Debian and Ubuntu ship it apart)
export UV_PYTHON_PREFERENCE=only-managed
# --inexact keeps anything else a maintainer installed in the same environment
if ! "$UV" sync --frozen --inexact; then
    echo "The libraries could not be installed: see the lines above."
    read -r -p "Press Enter to close. " _
    exit 1
fi
# detached, so this terminal window may close while the 11footy window stays open; what the
# window says goes to a log in the user's folder, shown here when it fails to start
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/11footy"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/launcher.log"
nohup .venv/bin/python -m desk >"$LOG" 2>&1 &
WINDOW=$!
sleep 3
if ! kill -0 "$WINDOW" 2>/dev/null; then
    wait "$WINDOW"
    CODE=$?
    if [ "$CODE" -ne 0 ]; then
        echo "The 11footy window could not start (exit $CODE). What it said:"
        cat "$LOG"
        echo "This is kept in $LOG."
        read -r -p "Press Enter to close. " _
        exit 1
    fi
fi
exit 0
