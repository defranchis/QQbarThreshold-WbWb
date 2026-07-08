#!/bin/bash
# HTCondor wrapper for the G_F-CONSISTENCY TEST grid (one MoCaNLO 4-run point).
#
# Runs the 'gf' EW scheme but with G_F TUNED so the derived α_Gμ equals the
# alphaz scheme's hard-EW α(M_Z) (Recola alZ = 0.0077983287 = 1/128.233).  With
# the α VALUE matched across the two schemes, the cross-fit gf(G_F_adj)↔alphaz
# isolates the PURE renormalisation-prescription residual (gfermi vs alphaZ),
# with the spurious input-α-value difference removed — the user's "last item".
#
# G_F_adj = 1.203449e-5 (+3.2 % vs default 1.1663787e-5): VERIFIED on 2026-06-17
# via Recola get_alpha_rcl → alpha_Gf = 0.0077983389 (matches alphaz α(M_Z) to
# 0.0013 %; the residual is card :.6g rounding → 1.20345e-5).
#
# Output goes to a DEDICATED EOS dir (results_gf_consistency), NEVER the
# production results dir — so it cannot overwrite the production gf grid.
#
# Args:  CHANNEL VARPOINT ECM        (varpoint is always 'nominal' here)
set -eo pipefail

CHANNEL="$1"; VARPOINT="$2"; ECM="$3"

ROOT=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold
REPO="$ROOT/WW_threshold"

GF_ADJ=1.203449e-5
OUTDIR=/eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results_gf_consistency

# shellcheck disable=SC1091
source "$ROOT/mocanlo/mocanlo_env.sh"
kinit -R 2>/dev/null || true

# Production statistics (500k events, 1.0 % precision, varpoint-correlated
# base-seed 1000), matching the existing scheme grids so the line shape is at
# the same MC quality.  5h/sub-run × 4 = 20h < the "tomorrow" (24h) flavour.
exec python3 -u "$REPO/scripts/indep_mocanlo/run_point.py" \
    "$CHANNEL" "$VARPOINT" "$ECM" \
    --scheme-alpha gf --fermi-constant "$GF_ADJ" \
    --precision-pct 1.0 --events 500000 --base-seed 1000 \
    --time-wall 0-05:00 --outdir "$OUTDIR"
