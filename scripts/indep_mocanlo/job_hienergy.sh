#!/bin/bash
# HTCondor wrapper for ONE above-threshold MoCaNLO point for the 157-365 GeV
# xsec curve + the 240 GeV anchor.  Writes to a SEPARATE results dir
# (results_hienergy) so the production threshold grid / headline fit is untouched.
#
# Args:  CHANNEL VARPOINT ECM [PREC] [EVENTS] [SCHEME]
set -eo pipefail

CHANNEL="$1"; VARPOINT="$2"; ECM="$3"
PREC="${4:-1.0}"; EVENTS="${5:-100000}"; SCHEME="${6:-gf}"

ROOT=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold
REPO="$ROOT/WW_threshold"

# shellcheck disable=SC1091
source "$ROOT/mocanlo/mocanlo_env.sh"
kinit -R 2>/dev/null || true

exec python3 -u "$REPO/scripts/indep_mocanlo/run_point.py" \
    "$CHANNEL" "$VARPOINT" "$ECM" \
    --scheme-alpha "$SCHEME" --precision-pct "$PREC" --events "$EVENTS" \
    --time-wall 0-05:00 \
    --outdir /eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results_hienergy
