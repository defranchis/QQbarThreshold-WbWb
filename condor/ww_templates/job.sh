#!/bin/bash
# HTCondor wrapper for a single WW NLL template (compute_xsec_ww.py
# sliced to one tag at one BEC shift). Pins the eMELA √s parallel
# dispatch to request_cpus via WW_ISR_NJOBS.
#
# Args:
#   $1  TAG          PARAMETERS tag from the live card (nominal|pseudodata|<param>_var|cross_mass_width)
#   $2  BEC_SHIFT    BEC shift in MeV (0 = nominal set; ±10/±30 = scan_{p,m}{10,30})

set -eo pipefail

TAG="$1"
BEC_SHIFT="$2"

REPO=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold

# QQbar_threshold install paths (libeMELApy.so etc.).
# shellcheck disable=SC1091
source "$REPO/setup.sh"

# Pin n_jobs for sigma_ISR_2leg_convolution to the slot's CPU allocation.
# Matches request_cpus = 4 in the submit file.
export WW_ISR_NJOBS=4

# Renew Kerberos token (best-effort).
kinit -R 2>/dev/null || true

cd "$REPO"

exec python3 -u compute_xsec_ww.py \
    --only-tag "$TAG" \
    --only-bec-shift "$BEC_SHIFT"
