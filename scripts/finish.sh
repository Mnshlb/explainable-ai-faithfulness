#!/usr/bin/env bash
# Wait for the main run to finish, then repair the glasses LIME pass
# (it failed on an incompatible kwarg), re-summarise and rebuild the report.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python

# Wait on the orchestrator, not on xai.run: that briefly disappears between
# tasks and the redo would otherwise start alongside the next one.
while pgrep -f "run_all\.sh" >/dev/null; do sleep 30; done
echo "=========== glasses LIME redo  $(date '+%H:%M:%S') ==========="
$PY -u -m xai.run --task glasses --methods lime --redo 2>&1 \
  | grep -viE "FutureWarning|warnings\.warn"
echo "=========== summarize  $(date '+%H:%M:%S') ==========="
$PY -u -m xai.summarize --all 2>&1 | grep -viE "FutureWarning|ConstantInput|  rho ="
echo "=========== report  $(date '+%H:%M:%S') ==========="
$PY -u -m xai.report
echo "=========== finished  $(date '+%H:%M:%S') ==========="
