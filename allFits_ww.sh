#!/usr/bin/env bash
# Full WW diagnostic suite — mirror of allFits_wbwb.sh with the scans
# WW actually supports. No --scaleVarsScan (WW card has no scale-
# variation templates), no --truevaluescan / --yukawaThScan /
# --widthscan / --shiftScan (WbWb-specific). doFit_ww.py also doesn't
# currently fan scans in parallel — they run sequentially inside each
# invocation.
#
# Placeholder: doFit_ww.py needs real cross-section templates in
# cards/ww_default.py:INPUT_DIRS["nominal"] and a functioning
# WWGenerator at process/ww/generator.py before producing meaningful
# output.
#
# Run from the WW_threshold/ directory after sourcing setup.sh.
set -euo pipefail

python3 doFit_ww.py --pseudo --systTable

python3 doFit_ww.py --systTable \
    --LSscan --lumiscans --alphaSscan \
    --BECscans --BESscans --chi2scans
