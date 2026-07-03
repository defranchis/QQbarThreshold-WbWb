#!/usr/bin/env bash
# Overnight Task 8 wrapper: run allFits_ww.sh against the COMMITTED (v3)
# templates while the working tree carries the uncommitted edge-aware
# quadrature (whose _RADIATOR_DISK_VERSION=4 bump makes fits fail-closed).
#
# The fits only READ template CSVs, so flipping the version constant back to
# 3 for the duration reproduces the pre-edit sweep byte-for-byte; the trap
# restores 4 no matter how the sweep exits. compute_xsec_ww.py then sees
# current templates and reuses them (no regen without sign-off).
set -uo pipefail

ISR=framework/process/ww/xsec_calculator/isr.py

restore() {
    sed -i 's/^_RADIATOR_DISK_VERSION = 3   # TEMP-SWEEP/_RADIATOR_DISK_VERSION = 4   # v4: edge-aware 2-leg quadrature (2026-07-03) —/' "$ISR"
    grep -q "_RADIATOR_DISK_VERSION = 4" "$ISR" && echo "[wrapper] version restored to 4"
}
trap restore EXIT

sed -i 's/^_RADIATOR_DISK_VERSION = 4   # v4: edge-aware 2-leg quadrature (2026-07-03) —/_RADIATOR_DISK_VERSION = 3   # TEMP-SWEEP/' "$ISR"
grep -q "_RADIATOR_DISK_VERSION = 3" "$ISR" || { echo "[wrapper] flip failed"; exit 1; }
echo "[wrapper] version flipped to 3 for the sweep"

time bash allFits_ww.sh
rc=$?
echo "[wrapper] allFits_ww.sh exit code: $rc"
exit $rc
