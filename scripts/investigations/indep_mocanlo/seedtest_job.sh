#!/bin/bash
# HTCondor wrapper for one seed-A/B test point (disconnect-safe, unlike ironic).
# Args: MODE(corr|decorr) VARPOINT ECM
set -eo pipefail
MODE="$1"; VP="$2"; ECM="$3"
ROOT=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold
REPO="$ROOT/WW_threshold"
# shellcheck disable=SC1091
source "$ROOT/mocanlo/mocanlo_env.sh"
kinit -R 2>/dev/null || true
FLAG=""; [ "$MODE" = decorr ] && FLAG="--decorrelate"
exec python3 -u "$REPO/scripts/indep_mocanlo/run_point.py" lnuqq "$VP" "$ECM" \
  --events 50000 --precision-pct 1 --time-wall 0-00:40 \
  --outdir "$ROOT/mocanlo/grid_gen/seedtest_results" \
  --workdir "$ROOT/mocanlo/grid_gen/seedtest_work" $FLAG
