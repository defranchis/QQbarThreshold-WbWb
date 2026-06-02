#!/bin/bash
# HTCondor wrapper for one independent-WW partonic grid point (MoCaNLO 4-run).
#
# Args:  CHANNEL  VARPOINT  ECM  [PRECISION_PCT]  [EVENTS]  [SCHEME_ALPHA]  [LEPTON_CUT]
#   CHANNEL    enuqq|lnuqq|qqqq|enuenu|emu|lnulnu
#   VARPOINT   nominal|massUp|massDn|widthUp|widthDn|cross
#   ECM        partonic √ŝ in GeV
#   LEPTON_CUT |cosθ_l|<COS fiducial cut (e.g. 0.95); "none" = inclusive
set -eo pipefail

CHANNEL="$1"; VARPOINT="$2"; ECM="$3"
PREC="${4:-0.5}"; EVENTS="${5:-200000}"; SCHEME="${6:-gf}"; LEPCUT="${7:-none}"

ROOT=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold
REPO="$ROOT/WW_threshold"

# shellcheck disable=SC1091
source "$ROOT/mocanlo/mocanlo_env.sh"
kinit -R 2>/dev/null || true

CUT_ARGS=()
[ "$LEPCUT" != "none" ] && CUT_ARGS=(--lepton-cut "$LEPCUT")

exec python3 -u "$REPO/scripts/indep_mocanlo/run_point.py" \
    "$CHANNEL" "$VARPOINT" "$ECM" \
    --scheme-alpha "$SCHEME" --precision-pct "$PREC" --events "$EVENTS" \
    --time-wall 0-08:00 "${CUT_ARGS[@]}"
