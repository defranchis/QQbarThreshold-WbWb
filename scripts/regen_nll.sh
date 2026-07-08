#!/bin/bash
# Regenerate the WW NLL production template set after the parametric-prior
# card batch (aem_isr nuisance, PEAK_ECM, lumi prior). Run on fcc-ironic.
set -e
cd /afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold
source setup.sh >/dev/null 2>&1
export PYTHONPATH="$PWD:$PYTHONPATH"
# Inner eMELA pool: 48-way fork made all workers first-touch the WHIZARD morph
# grid on AFS at once → "Connection timed out". Pre-warm the AFS client cache,
# then run with a gentler concurrency (16) so the per-worker reads hit cache.
export WW_ISR_NJOBS="${WW_ISR_NJOBS:-16}"
GRID=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard/work/grid_fine/grid.csv
cat "$GRID" >/dev/null 2>&1 || true
rm -f output_xsec/ww/nominal/WW_*.txt output_xsec/ww/BEC/scan_*/WW_*.txt
set +e
python3 compute_xsec_ww.py --force
echo "REGEN_EXIT=$?"
