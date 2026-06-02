#!/bin/bash
# A/B test of correlated vs decorrelated MC seeds for the morph response.
# lnuqq, varpoints {nominal, mp10, mm10}, several √ŝ, 50k events, both seed modes,
# run in parallel on ironic into a dedicated seedtest dir (does NOT touch the live
# campaign results). Then seedtest_compare.py quantifies the de-noise gain.
ROOT=/afs/cern.ch/work/m/mdefranc/private/FCC/QQbar_threshold
REPO="$ROOT/WW_threshold"
RP="$REPO/scripts/indep_mocanlo/run_point.py"
OUT="$ROOT/mocanlo/grid_gen/seedtest_results"
WORK="$ROOT/mocanlo/grid_gen/seedtest_work"
# shellcheck disable=SC1091
source "$ROOT/mocanlo/mocanlo_env.sh"
kinit -R 2>/dev/null || true
mkdir -p "$OUT" "$WORK"

EVENTS=50000
ECMS="159.0 160.0 161.0 162.0 163.0"
VPS="nominal mp10 mm10"
echo "host=$(hostname) start=$(date)"

pids=()
for mode in corr decorr; do
  FLAG=""; [ "$mode" = decorr ] && FLAG="--decorrelate"
  for vp in $VPS; do
    for ecm in $ECMS; do
      python3 "$RP" lnuqq "$vp" "$ecm" --events "$EVENTS" --precision-pct 1 \
        --time-wall 0-00:40 --outdir "$OUT" --workdir "$WORK" $FLAG \
        > "$OUT/log_${mode}_${vp}_${ecm}.txt" 2>&1 &
      pids+=($!)
    done
  done
done
echo "launched ${#pids[@]} runs in parallel"
wait
echo "ALL_DONE=$(date)"
