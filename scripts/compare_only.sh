#!/bin/bash
# Re-run ONLY the scenario comparison (doFit_ww.py --compareScenarios) on
# fcc-ironic. The theory-ladder templates are reused from the persistent cache
# (card fingerprint unchanged), so this is light; it regenerates the scenario
# text/CSV plus all scenario_compare_* plots (layout, ellipses, per-POI syst
# bars, and the per-scenario systematic sweeps). cd is baked in so the remote
# shell (which starts in $HOME) finds setup.sh — see memory feedback_ssh_remote_cd.
set +e
cd /afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold
source setup.sh >/dev/null 2>&1
export PYTHONPATH="$PWD:$PYTHONPATH"
cat "$PWD/../whizard/work/grid_fine/grid.csv" >/dev/null 2>&1 || true   # pre-warm AFS grid
WW_ISR_NJOBS=16 python3 doFit_ww.py --compareScenarios --ladderWorkers 32
echo "COMPARE_EXIT=$?"
