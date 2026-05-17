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
# Run from the WW_threshold/ directory after sourcing setup.sh.
set -euo pipefail

python doFit_wbwb.py --scaleVars --scaleVarsScan

python doFit_wbwb.py --pseudo --systTable

python doFit_wbwb.py --systTable \
    --LSscan --lumiscans --alphaSscan \
    --BECscans --BESscans --chi2scans
