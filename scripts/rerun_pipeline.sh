#!/bin/bash
# Full WW re-run pipeline after the parametric-prior card batch. Run on
# fcc-ironic. Produces the numeric outputs the report/paper tables consume:
#   - systematics table (SYST_TABLE_PATH)         [light: reads NLL templates]
#   - channel extrapolation (plots/channel_extrap/) [light]
#   - theory ladder (plots/theory_ladder/)        [heavy: regenerates rung templates]
#   - scenario comparison (plots/scenario_compare/) [reuses ladder cache]
# Pre-warms the WHIZARD morph grid and uses a gentle worker count to avoid the
# AFS fork-storm timeout (see memory feedback_afs_grid_contention).
set +e
cd /afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold
source setup.sh >/dev/null 2>&1
export PYTHONPATH="$PWD:$PYTHONPATH"
GRID=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard/work/grid_fine/grid.csv
cat "$GRID" >/dev/null 2>&1 || true

echo "===== [1/4] baseline fit + systTable ====="
python3 doFit_ww.py --systTable --noPlots
echo "SYSTTABLE_EXIT=$?"

echo "===== [2/4] channel extrapolation ====="
python3 doFit_ww.py --channelExtrap
echo "CHANEXTRAP_EXIT=$?"

echo "===== [3/4] theory ladder ====="
python3 doFit_ww.py --theoryLadder --ladderWorkers 32
echo "LADDER_EXIT=$?"

echo "===== [4/4] scenario comparison ====="
python3 doFit_ww.py --compareScenarios --ladderWorkers 32
echo "SCENARIOS_EXIT=$?"

echo "===== PIPELINE DONE ====="
