#!/bin/bash
# Run one WHIZARD-ISR scan WITHOUT the sqrts_hat>155 GeV cut, fresh test.
# Args:
#   $1  LABEL    short identifier ("alpha0_nocut" / "alphaGmu_nocut")
#   $2  ALPHA    numeric value of isr_alpha
set -eo pipefail
LABEL="$1"
ALPHA="$2"

WW_TOP=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/WW_threshold
WHIZARD_TOP=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold/whizard
INV_DIR="$WW_TOP/scripts/investigations/whizard_isr_verification"
OUT_DIR="$INV_DIR/work_$LABEL"

mkdir -p "$OUT_DIR"

# shellcheck disable=SC1091
source "$WHIZARD_TOP/setup.sh"
export OMP_NUM_THREADS=8

WORK=$(mktemp -d -t whiz_isr_${LABEL}.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
cd "$WORK"

sed "s/\$ALPHA/$ALPHA/" "$INV_DIR/job_isr_nocut.sin.tmpl" > job.sin

RC=0
whizard job.sin > whizard.log 2>&1 || RC=$?

{
    echo "# sqrts_GeV sigma_fb err_fb"
    grep '^RESULT ' whizard.log 2>/dev/null | sed 's/^RESULT //' | awk '!seen[$0]++' || true
} > results.csv

cp job.sin whizard.log results.csv "$OUT_DIR/"
echo "[$LABEL] done with rc=$RC, results in $OUT_DIR"
exit "$RC"
