#!/bin/bash
# Run WHIZARD no-ISR Born scan to confirm process definition matches BFS.
set -eo pipefail

WW_TOP=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold
WHIZARD_TOP=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard
INV_DIR="$WW_TOP/scripts/investigations/whizard_isr_verification"
OUT_DIR="$INV_DIR/work_born"

mkdir -p "$OUT_DIR"
# shellcheck disable=SC1091
source "$WHIZARD_TOP/setup.sh"
export OMP_NUM_THREADS=8

WORK=$(mktemp -d -t whiz_born.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

cp "$INV_DIR/job_born.sin" job.sin

RC=0
whizard job.sin > whizard.log 2>&1 || RC=$?

{
    echo "# sqrts_GeV sigma_fb err_fb"
    grep '^RESULT ' whizard.log 2>/dev/null | sed 's/^RESULT //' | awk '!seen[$0]++' || true
} > results.csv

cp job.sin whizard.log results.csv "$OUT_DIR/"
echo "[born] done with rc=$RC, results in $OUT_DIR"
exit "$RC"
