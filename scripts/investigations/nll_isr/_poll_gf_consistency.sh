#!/bin/bash
# Background poll: wait for condor cluster 12923949 (gf-consistency grid) to drain,
# then run the cross-fit.  Best-effort; refreshes the Kerberos token each loop.
set +e
cd /afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold || exit 1
CLUSTER=12923949
LOG=/tmp/gfcons_crossfit_result.log
for i in $(seq 1 240); do          # 240 × 240s ≈ 16 h cap
    kinit -R 2>/dev/null || true
    aklog 2>/dev/null || true
    N=$(condor_q "$CLUSTER" -totals 2>/dev/null | grep -oP 'Total for query: \K[0-9]+' | head -1)
    [ -z "$N" ] && N=$(condor_q "$CLUSTER" 2>/dev/null | grep -cE "^ *${CLUSTER}\.")
    NFILES=$(ls /eos/user/m/mdefranc/FCC/QQbar_threshold/grid_gen/results_gf_consistency/*.csv 2>/dev/null | wc -l)
    echo "[$(date +%H:%M)] iter $i: queue=$N csv=$NFILES/99"
    if [ "${N:-1}" = "0" ]; then
        echo "queue drained — running cross-fit"
        source setup.sh
        WW_INDEP_NJOBS=16 PYTHONPATH=$PWD:$PYTHONPATH \
            python3 scripts/investigations/nll_isr/gf_consistency_crossfit.py > "$LOG" 2>&1
        echo "CROSSFIT_EXIT=$?"
        break
    fi
    sleep 240
done
echo "poll done"
