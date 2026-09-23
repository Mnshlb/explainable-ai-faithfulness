#!/usr/bin/env bash
# Full pipeline: explain every prediction of every task, verify nothing is
# missing, aggregate, and regenerate the report.
#
# Safe to interrupt and re-run at any point. Each task appends to its own
# JSONL and resumes from it; only the glasses LIME repair rewrites in place,
# and it does so through an atomic swap.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PY=.venv/bin/python
QUIET='FutureWarning|warnings\.warn|Loading weights|^Key |UNEXPECTED|MISSING|^Notes:|^- |^\[transformers\]|^Warning:|^-----|ConstantInput|  rho ='

step () { echo; echo "======== $1  $(date '+%F %H:%M:%S') ========"; }

# The first glasses pass lost LIME to an incompatible kwarg. Recompute just
# that method over the existing records rather than discarding Grad-CAM and
# SHAP, which are already complete.
step "glasses — repair LIME"
$PY -u -m xai.run --task glasses --methods lime --redo 2>&1 | grep -viE "$QUIET"

step "tweet — explain every prediction"
$PY -u -m xai.run --task tweet 2>&1 | grep -viE "$QUIET"

step "movie — explain every prediction"
$PY -u -m xai.run --task movie 2>&1 | grep -viE "$QUIET"

step "verify completeness"
$PY -u -m xai.verify --all 2>&1 | grep -viE "$QUIET"

step "summarize"
$PY -u -m xai.summarize --all 2>&1 | grep -viE "$QUIET"

# The four analyses below are what turn a description of the explanations into a
# comparison of them. Each writes its own artifact and merges rather than
# overwriting, so any one can be re-run alone.
step "model cards — architecture, training provenance, accuracy on every split"
$PY -u -m xai.modelcard --task all 2>&1 | grep -viE "$QUIET"

step "plausibility — against the human annotation shipped with the datasets"
$PY -u -m xai.evaluate --task all 2>&1 | grep -viE "$QUIET"

step "faithfulness — deletion, against a random-ranking control"
$PY -u -m xai.faithfulness --task all 2>&1 | grep -viE "$QUIET"

step "behaviour — what each method names, and the budget analysis"
$PY -u -m xai.behaviour --task all 2>&1 | grep -viE "$QUIET"

step "report — HTML intermediate, then PDF"
$PY -u -m xai.report 2>&1 | grep -viE "$QUIET"

step "finished"
