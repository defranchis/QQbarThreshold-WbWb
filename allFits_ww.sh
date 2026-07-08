#!/usr/bin/env bash
# Full WW diagnostic suite — mirror of allFits_wbwb.sh with the scans
# WW actually supports. No --scaleVarsScan (WW card has no scale-
# variation templates), no --truevaluescan / --yukawaThScan /
# --widthscan / --shiftScan (WbWb-specific). The requested scans fan
# out in parallel by default (doFit_ww.py --parallel N, default 6;
# pass --parallel 1 to force sequential).
#
# Pre-req: run `python3 compute_xsec_ww.py` once to populate the
# nominal + BEC-variation templates in cards/ww_default.py:INPUT_DIRS.
# Run from the WW_threshold/ directory.
set -euo pipefail

python3 compute_xsec_ww.py    # regenerate templates (idempotent, ~1 s)

python3 doFit_ww.py --pseudo --systTable

python3 doFit_ww.py --systTable \
    --LSscan --lumiscans --alphaSscan \
    --BECscans --BESscans --chi2scans
