#!/usr/bin/env bash
# Resume the pipeline: glasses is already complete and verified, so pick up
# with the interrupted tweet pass, then movie, verify, summarize, report.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
QUIET='FutureWarning|warnings\.warn|Loading weights|^Key |UNEXPECTED|MISSING|^Notes:|^- |^\[transformers\]|^Warning:|^-----|ConstantInput|  rho ='

step () { echo; echo "======== $1  $(date '+%F %H:%M:%S') ========"; }

step "tweet — explain every prediction (resuming)"
$PY -u -m xai.run --task tweet 2>&1 | grep -viE "$QUIET"

step "movie — explain every prediction"
$PY -u -m xai.run --task movie 2>&1 | grep -viE "$QUIET"

step "verify completeness"
$PY -u -m xai.verify --all 2>&1 | grep -viE "$QUIET"

step "summarize"
$PY -u -m xai.summarize --all 2>&1 | grep -viE "$QUIET"

step "report"
$PY -u -m xai.report 2>&1 | grep -viE "$QUIET"

step "finished"
