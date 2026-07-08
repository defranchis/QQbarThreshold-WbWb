#!/usr/bin/env bash
# Full WbWb diagnostic suite, collapsed from the legacy 10-line
# allFits.sh into 3 invocations:
#
#  1. --scaleVarsScan first — needs --scaleVars, which is mutually
#     exclusive with --BECscans / --BESscans per doFit_wbwb._check_args.
#     Its --scaleVars setup also rewrites plot_parameter_variations.pdf
#     / fit_scenario.pdf from a different template tag; running it
#     first means the Asimov invocation below overwrites them back to
#     the nominal version.
#  2. --pseudo --systTable — writes a pseudo-stat systematics_table.tex,
#     overwritten by the Asimov invocation below.
#  3. All Asimov scans in parallel + final Asimov --systTable.
#     doFit_wbwb.py fans the scan jobs across cores via --parallel 6
#     (override with --parallel N).
#
# Run from the WW_threshold/ directory.
set -euo pipefail

# WbWb pulls in xsec_calculator.xsec_calc (pybind11 → libQQbar_threshold.so).
# That .so has no embedded RPATH, so the dynamic loader needs
# LD_LIBRARY_PATH pointing at install/lib — sourced from setup.sh.
# (WW takes a different code path and does not need this; see allFits_ww.sh.)
source "$(dirname "$0")/setup.sh"

python3 doFit_wbwb.py --parallel 6 --scaleVars --scaleVarsScan

python3 doFit_wbwb.py --pseudo --systTable

python3 doFit_wbwb.py --parallel 6 --systTable \
    --LSscan --lumiscans --alphaSscan \
    --BECscans --BESscans --chi2scans
